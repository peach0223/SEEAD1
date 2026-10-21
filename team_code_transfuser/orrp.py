"""Candidate ranking and conservative swept-box collision checks.

Coordinates are virtual LiDAR x-forward/y-right with ego at (-lidar_x, 0).
Rotation intervals use midpoint rectangles inflated by a corner-displacement
bound. Neighbor motion uses constant velocity and fixed yaw.
"""

import math

try:
    from .vehicle_control import forward_reachable
except ImportError:
    from vehicle_control import forward_reachable

import torch
from torch import nn
from torch.nn import functional as F


def _parameter(config, name, default):
    return getattr(config, name, default)


def lbc_target_to_ego(target_point):
    return torch.stack((-target_point[..., 1], target_point[..., 0]), dim=-1)


def wrap_angle(angle):

    return torch.atan2(torch.sin(angle), torch.cos(angle))


def trajectory_kinematics(waypoints, ego_vel, dt, lidar_x=1.3):


    if dt <= 0:
        raise ValueError("The waypoint interval must be positive")
    origin = waypoints.new_tensor([-lidar_x, 0.0])
    initial = origin.expand(*waypoints.shape[:-2], 1, 2)
    steps = torch.diff(torch.cat((initial, waypoints), dim=-2), dim=-2)
    distances = torch.linalg.vector_norm(steps, dim=-1)
    speeds = distances / dt
    speed0 = ego_vel.reshape(waypoints.shape[0], -1)[:, :1].to(waypoints)
    speed0 = speed0[:, None].expand(-1, waypoints.shape[1], -1)
    accelerations = torch.diff(torch.cat((speed0, speeds), dim=-1), dim=-1) / dt
    moving = distances > 1e-6
    safe_x = torch.where(moving, steps[..., 0], torch.ones_like(distances))
    safe_y = torch.where(moving, steps[..., 1], torch.zeros_like(distances))
    raw_headings = torch.atan2(safe_y, safe_x)
    previous = torch.zeros_like(raw_headings[..., 0])
    headings = []
    for step in range(waypoints.shape[-2]):
        previous = torch.where(moving[..., step], raw_headings[..., step], previous)
        headings.append(previous)
    headings = torch.stack(headings, dim=-1)
    heading0 = torch.zeros_like(headings[..., :1])
    heading_rates = wrap_angle(torch.diff(torch.cat((heading0, headings), dim=-1), dim=-1)) / dt
    return distances, accelerations, heading_rates, headings


def detection_neighbors(labels, config, pred_len=None, return_context=False, return_geometry=False):


    if labels.ndim != 3 or labels.shape[-1] not in (7, 8):
        raise ValueError("labels must have shape (B, M, 7) or (B, M, 8)")
    pred_len = int(pred_len if pred_len is not None else _parameter(config, "pred_len", 4))
    dt = float(_parameter(config, "bev_waypoint_dt", 0.5))
    ppm = float(_parameter(config, "pixels_per_meter", 8.0))
    if dt <= 0 or ppm <= 0 or pred_len < 1:
        raise ValueError("pred_len, pixels_per_meter and bev_waypoint_dt must be positive")
    if not labels.is_floating_point():
        labels = labels.float()
    valid = torch.isfinite(labels).all(dim=-1) & (labels[..., 2] > 0) & (labels[..., 3] > 0)
    if labels.shape[-1] == 8:
        valid = valid & (labels[..., 7] > float(_parameter(config, "bb_confidence_threshold", 0.3)))
    boxes = torch.where(valid[..., None], labels, torch.zeros_like(labels))
    height = float(_parameter(config, "lidar_resolution_height", 256))
    width = float(_parameter(config, "lidar_resolution_width", 256))
    centres = torch.stack(((height - boxes[..., 1]) / ppm,
                           (boxes[..., 0] - width / 2.0) / ppm), dim=-1)
    centres = torch.where(valid[..., None], centres, torch.zeros_like(centres))
    velocity = boxes[..., 5, None] * torch.stack((torch.cos(boxes[..., 4]),
                                                 torch.sin(boxes[..., 4])), dim=-1)
    times = torch.arange(1, pred_len + 1, device=labels.device, dtype=labels.dtype) * dt
    trajectories = centres[..., None, :] + times[None, None, :, None] * velocity[..., None, :]
    mask = valid[..., None].expand(-1, -1, pred_len)
    trajectories = torch.where(mask[..., None], trajectories, torch.zeros_like(trajectories))
    if return_context or return_geometry:
        # Raster width is lateral, raster height is longitudinal. Do not swap
        # physical half-length and half-width when converting detector boxes.
        extents = boxes[..., [3, 2]] / (2.0 * ppm)
        radii = torch.linalg.vector_norm(extents, dim=-1)
        context = (trajectories, mask, centres, valid, radii)
        if return_geometry:
            return context + (extents, boxes[..., 4])
        return context
    return trajectories, mask


class ORRPPlanner(nn.Module):


    def __init__(self, config, channels, fused_dim=512):
        super().__init__()
        self.config = config
        self.pred_len = int(_parameter(config, "pred_len", 4))
        self.num_candidates = int(_parameter(config, "bev_num_candidates", 6))
        self.hidden_size = int(_parameter(config, "bev_orrp_hidden", 64))
        self.dt = float(_parameter(config, "bev_waypoint_dt", 0.5))
        self.lidar_x = float(_parameter(config, "lidar_pos", [1.3, 0.0, 2.5])[0])
        self.cost_floor = float(_parameter(config, "bev_orrp_epsilon", 0.1))
        self.collision_radius = float(_parameter(config, "bev_orrp_collision_radius", 2.5))
        self.attention_grid = int(_parameter(config, "bev_orrp_attention_grid", 8))
        self.selection = _parameter(config, "bev_score_selection", "learned")
        self.objective = _parameter(config, 'bev_score_objective', 'ratio')
        self.all_candidate_weight = float(_parameter(config, "bev_all_candidate_weight", 0.05))
        self.ego_extent = (float(_parameter(config, "ego_extent_x", 2.4508416652679443)),
                           float(_parameter(config, "ego_extent_y", 1.0641621351242065)))
        self.safety_margin = float(_parameter(config, "bev_collision_margin", 0.3))
        self.risk_scale = float(_parameter(config, "bev_orrp_risk_scale", 2.0))
        self.unsafe_penalty = float(_parameter(config, "bev_orrp_unsafe_penalty", 5.0))
        self.score_bound = float(_parameter(config, "bev_orrp_score_bound", 100.0))
        self.rotation_step = float(_parameter(config, 'bev_rotation_step', math.pi / 36))
        if any(not math.isfinite(v) or v <= 0 for v in
               (*self.ego_extent, self.dt, self.cost_floor, self.collision_radius,
                self.risk_scale, self.score_bound, self.rotation_step)):
            raise ValueError("Physical dimensions, dt, risk scale and score bound must be finite and positive")
        if any(not math.isfinite(v) or v < 0 for v in
               (self.safety_margin, self.unsafe_penalty, self.all_candidate_weight, *self._weights().values())):
            raise ValueError("ORRP weights and safety margin must be finite and nonnegative")
        if self.objective not in ("ratio", "cost", "reward"):
            raise ValueError("bev_score_objective must be ratio, cost or reward")
        if min(self.pred_len, self.num_candidates, self.hidden_size, self.attention_grid) < 1:
            raise ValueError("Trajectory and attention dimensions must be positive")
        if min(self.dt, self.cost_floor, self.collision_radius) <= 0:
            raise ValueError("Waypoint dt, control cost floor and collision radius must be positive")
        if self.selection not in ("learned", "analytic"):
            raise ValueError("bev_score_selection must be 'learned' or 'analytic'")
        hidden = self.hidden_size
        self.context = nn.Sequential(nn.Linear(fused_dim + channels + 3, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden))
        self.mode_embeddings = nn.Embedding(self.num_candidates, hidden)
        self.decoder = nn.GRUCell(5, hidden)
        self.waypoint_delta = nn.Linear(hidden, 2)
        self.bev_projection = nn.Conv2d(channels, hidden, 1)
        self.bev_position = nn.Linear(2, hidden)
        self.trajectory_encoder = nn.Sequential(nn.Linear(5, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        heads = next(heads for heads in (4, 2, 1) if hidden % heads == 0)
        self.cross_attention = nn.MultiheadAttention(hidden, heads, dropout=0.0, batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden)
        self.score_head = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def _weights(self):
        return {name: float(_parameter(self.config, "bev_orrp_" + name, 1.0))
                for name in ("eta1", "eta2", "mu1", "mu2", "rho1", "rho2")}

    def _semantic_similarity(self, trajectories, target_point, mask=None):


        origin = trajectories.new_tensor([-self.lidar_x, 0.0])
        encoded = trajectories - origin
        fraction = torch.arange(1, self.pred_len + 1, device=trajectories.device,
                                dtype=trajectories.dtype) / self.pred_len

        goal = target_point[:, None, None, :] * fraction[None, None, :, None]
        goal = goal.expand_as(encoded)
        if mask is not None:
            encoded = torch.where(mask[..., None], encoded, torch.zeros_like(encoded))
            goal = torch.where(mask[..., None], goal, torch.zeros_like(goal))
        cosine = F.cosine_similarity(encoded.flatten(-2), goal.flatten(-2), dim=-1, eps=1e-6)
        return ((cosine + 1.0) * 0.5).clamp(0.0, 1.0)

    @staticmethod
    def _sat_collision(offset, motion, radii):
        """Exact overlap interval for translating fixed-yaw rectangles."""
        moving = motion.abs() > torch.finfo(motion.dtype).eps
        denominator = torch.where(moving, motion, torch.ones_like(motion))
        first, second = (-radii - offset) / denominator, (radii - offset) / denominator
        lower = torch.where(moving, torch.minimum(first, second), -torch.inf)
        upper = torch.where(moving, torch.maximum(first, second), torch.inf)
        separated = (~moving & (offset.abs() > radii)).any(-1)
        return (lower.amax(-1).clamp_min(0) <= upper.amin(-1).clamp_max(1)) & ~separated

    @staticmethod
    def _minimum_clearance(offset, motion, radii):
        """Minimum SAT separation over [0,1], a lower bound on Euclidean clearance.

        The maximum of eight affine functions is minimized at an endpoint or
        line intersection. Looping 28 pairs avoids a large pairwise tensor.
        """
        intercept = torch.cat((offset - radii, -offset - radii), -1)
        slope = torch.cat((motion, -motion), -1)
        best = torch.minimum(intercept.amax(-1), (intercept + slope).amax(-1))
        eps = torch.finfo(offset.dtype).eps
        for first in range(8):
            for second in range(first + 1, 8):
                denominator = slope[..., first] - slope[..., second]
                valid = denominator.abs() > eps
                denominator = torch.where(valid, denominator, torch.ones_like(denominator))
                position = (intercept[..., second] - intercept[..., first]) / denominator
                valid &= (position >= 0) & (position <= 1)
                value = (intercept + position.clamp(0, 1)[..., None] * slope).amax(-1)
                best = torch.minimum(best, value.masked_fill(~valid, torch.inf))
        return best

    def _project_geometry(self, relative, motion, ego_yaw, extents, other_yaw):
        # Leading dimensions are B, candidate, neighbor, time.
        ego_u = torch.stack((ego_yaw.cos(), ego_yaw.sin()), -1)[:, :, None]
        ego_v = torch.stack((-ego_yaw.sin(), ego_yaw.cos()), -1)[:, :, None]
        other_u = torch.stack((other_yaw.cos(), other_yaw.sin()), -1)[:, None, :, None]
        other_v = torch.stack((-other_yaw.sin(), other_yaw.cos()), -1)[:, None, :, None]
        shape = relative.shape
        axes = torch.stack((ego_u.expand(shape), ego_v.expand(shape),
                            other_u.expand(shape), other_v.expand(shape)), -2)
        offset = (relative[..., None, :] * axes).sum(-1)
        projected_motion = (motion[..., None, :] * axes).sum(-1)
        radii = (self.ego_extent[0] * (axes * ego_u[..., None, :]).sum(-1).abs()
                 + self.ego_extent[1] * (axes * ego_v[..., None, :]).sum(-1).abs()
                 + extents[:, None, :, None, 0, None] * (axes * other_u[..., None, :]).sum(-1).abs()
                 + extents[:, None, :, None, 1, None] * (axes * other_v[..., None, :]).sum(-1).abs()
                 + self.safety_margin)
        return offset, projected_motion, radii

    @torch.no_grad()
    def _collision_geometry(self, candidates, neighbors, neighbor_mask, neighbor_current,
                            neighbor_current_mask, neighbor_radii,
                            neighbor_extents=None, neighbor_yaws=None, need_clearance=True, ego_headings=None):
        batch, modes = candidates.shape[:2]
        dtype = torch.float64 if candidates.dtype == torch.float64 else torch.float32
        candidates = candidates.to(dtype)
        if neighbors is None or neighbors.shape[1] == 0:
            return (torch.zeros((batch, modes, 0), device=candidates.device, dtype=torch.bool),
                    candidates.new_empty((batch, 0)), candidates.new_empty((batch, modes, 0)))
        if neighbor_current is None or neighbor_current_mask is None:
            raise ValueError('Continuous checking requires current neighbor positions and masks')
        horizon = candidates.shape[-2]
        if neighbors.ndim != 4 or neighbors.shape[0] != batch or neighbors.shape[2:] != (horizon, 2):
            raise ValueError('Invalid neighbor trajectory shape')
        if neighbor_current.shape != neighbors.shape[:2] + (2,):
            raise ValueError('Invalid current neighbor geometry shape')
        if neighbor_current_mask.shape != neighbors.shape[:2]:
            raise ValueError('Invalid current neighbor mask shape')
        if (neighbor_extents is None) != (neighbor_yaws is None):
            raise ValueError('Neighbor extents and yaws must be supplied together')
        if neighbor_extents is None:
            # Compatibility for old five-item contexts only. A radius cannot
            # recover an oriented rectangle: use its conservative square.
            if neighbor_radii is None or neighbor_radii.shape != neighbors.shape[:2]:
                raise ValueError('Continuous checking requires rectangular geometry or legacy radii')
            neighbor_extents = neighbor_radii[..., None].expand(-1, -1, 2)
            neighbor_yaws = torch.zeros_like(neighbor_radii)
        if neighbor_extents.shape != neighbors.shape[:2] + (2,) or neighbor_yaws.shape != neighbors.shape[:2]:
            raise ValueError('Invalid neighbor rectangle shape')
        current_valid = neighbor_current_mask.to(device=candidates.device, dtype=torch.bool)
        extents, yaws = neighbor_extents.to(candidates), neighbor_yaws.to(candidates)
        current = neighbor_current.to(candidates)
        finite_geometry = (torch.isfinite(current).all(-1) & torch.isfinite(extents).all(-1)
                           & (extents > 0).all(-1) & torch.isfinite(yaws))
        if (current_valid & ~finite_geometry).any():
            raise ValueError('Active neighbor rectangles must have finite positions/yaw and positive extents')
        extents = torch.where(current_valid[..., None], extents, torch.zeros_like(extents))
        yaws = torch.where(current_valid, yaws, torch.zeros_like(yaws))
        current = torch.where(current_valid[..., None], current, torch.zeros_like(current))
        neighbors = neighbors.to(candidates)
        future_valid = torch.isfinite(neighbors).all(-1)
        if neighbor_mask is not None:
            if neighbor_mask.shape != future_valid.shape:
                raise ValueError('Invalid future neighbor mask shape')
            future_valid &= neighbor_mask.to(device=candidates.device, dtype=torch.bool)
        future_valid &= current_valid[..., None]
        positions, last = [current], current
        for step in range(horizon):
            last = torch.where(future_valid[..., step, None], neighbors[..., step, :], last)
            positions.append(last)
        other = torch.stack(positions, -2)
        origin = candidates.new_tensor([-self.lidar_x, 0.0])
        ego = torch.cat((origin.expand(batch, modes, 1, 2), candidates), -2)
        headings = (trajectory_kinematics(candidates, candidates.new_zeros(batch), self.dt, self.lidar_x)[3]
                    if ego_headings is None else ego_headings.to(candidates))
        if headings.shape != candidates.shape[:-1] or not torch.isfinite(headings).all():
            raise ValueError('Invalid ego rollout headings')
        starts = torch.cat((torch.zeros_like(headings[..., :1]), headings[..., :-1]), -1)
        turns = wrap_angle(headings - starts)
        pieces = max(1, int(torch.ceil(turns.abs().max() / self.rotation_step).item()))
        u = torch.arange(pieces + 1, device=candidates.device, dtype=dtype) / pieces
        ego_points = ego[..., :-1, None, :] + (ego[..., 1:, :] - ego[..., :-1, :])[..., None, :] * u[None, None, None, :, None]
        other_points = other[..., :-1, None, :] + (other[..., 1:, :] - other[..., :-1, :])[..., None, :] * u[None, None, None, :, None]
        relative = other_points[:, None] - ego_points[:, :, None]
        begin = relative[..., :-1, :].flatten(-3, -2)
        motion = (relative[..., 1:, :] - relative[..., :-1, :]).flatten(-3, -2)
        midpoint_yaws = (starts[..., None] + turns[..., None] * ((u[:-1] + u[1:]) / 2)).flatten(-2)
        offset, projected_motion, radii = self._project_geometry(begin, motion, midpoint_yaws, extents, yaws)
        # Every corner stays within 2*r*sin(delta_yaw/4) of the midpoint box.
        rotational_error = 2 * math.hypot(*self.ego_extent) * torch.sin(turns.abs() / (4 * pieces))
        rotational_error = rotational_error[..., None].expand(-1, -1, -1, pieces).flatten(-2)
        radii = radii + rotational_error[:, :, None, :, None]
        collisions = self._sat_collision(offset, projected_motion, radii).any(-1) & current_valid[:, None]
        clearance = (self._minimum_clearance(offset, projected_motion, radii).amin(-1)
                     if need_clearance else candidates.new_full((batch, modes, neighbors.shape[1]), torch.inf))
        clearance = clearance.masked_fill(~current_valid[:, None], torch.inf)
        proximity = torch.linalg.vector_norm(current - origin, dim=-1).masked_fill(~current_valid, torch.inf)
        return collisions, proximity, clearance

    @torch.no_grad()
    def collision_state(self, candidates, neighbors, neighbor_mask, neighbor_current,
                        neighbor_current_mask, neighbor_radii=None,
                        neighbor_extents=None, neighbor_yaws=None):
        with torch.autocast(device_type=candidates.device.type, enabled=False):
            collisions, proximity, _ = self._collision_geometry(candidates, neighbors, neighbor_mask,
                neighbor_current, neighbor_current_mask, neighbor_radii, neighbor_extents, neighbor_yaws)
        return collisions, proximity

    @torch.no_grad()
    def compute_reference_scores(self, candidates, ego_vel, target_point=None,
                                 neighbors=None, neighbor_mask=None, neighbor_current=None,
                                 neighbor_current_mask=None, neighbor_radii=None,
                                 neighbor_extents=None, neighbor_yaws=None):
        with torch.autocast(device_type=candidates.device.type, enabled=False):
            return self._reference_impl(candidates, ego_vel, target_point, neighbors, neighbor_mask,
                neighbor_current, neighbor_current_mask, neighbor_radii, neighbor_extents, neighbor_yaws)

    def _reference_impl(self, candidates, ego_vel, target_point, neighbors, neighbor_mask,
                        neighbor_current, neighbor_current_mask, neighbor_radii, neighbor_extents, neighbor_yaws, geometry=None):
        if candidates.ndim != 4 or candidates.shape[-2:] != (self.pred_len, 2):
            raise ValueError("candidates must have shape (B, N, pred_len, 2)")
        dtype = torch.float64 if candidates.dtype == torch.float64 else torch.float32
        candidates = candidates.to(dtype)
        if not torch.isfinite(candidates).all() or not torch.isfinite(ego_vel).all():
            raise ValueError('Candidate trajectories and ego velocity must be finite')
        ego_vel = ego_vel.to(candidates)
        weights = self._weights()
        distances, accelerations, heading_rates, _ = trajectory_kinematics(
            candidates, ego_vel, self.dt, self.lidar_x)
        cost = (weights["eta1"] * accelerations.abs().sum(-1)
                + weights["eta2"] * heading_rates.abs().sum(-1)).clamp_min(self.cost_floor)
        reward = weights["mu1"] * distances.sum(-1)
        reference = -cost if self.objective == 'cost' else reward if self.objective == 'reward' else 1.0 + reward / cost
        collisions, proximity, clearance = geometry if geometry is not None else self._collision_geometry(candidates, neighbors, neighbor_mask,
            neighbor_current, neighbor_current_mask, neighbor_radii, neighbor_extents, neighbor_yaws)
        if self.objective != 'cost' and neighbors is not None and neighbors.shape[1] > 0:
            neighbors = neighbors.to(candidates)
            valid = torch.isfinite(neighbors).all(dim=-1)
            if neighbor_mask is not None:
                if neighbor_mask.shape != neighbors.shape[:-1]:
                    raise ValueError("neighbor_mask must have shape (B, M, pred_len)")
                valid &= neighbor_mask.to(device=neighbors.device, dtype=torch.bool)
            valid &= neighbor_current_mask.to(device=neighbors.device, dtype=torch.bool)[..., None]
            neighbors = torch.where(valid[..., None], neighbors, torch.zeros_like(neighbors))
            target = (candidates.new_tensor([1.0, 0.0]).expand(candidates.shape[0], -1)
                      if target_point is None else lbc_target_to_ego(target_point.to(candidates)))
            if not torch.isfinite(target).all():
                raise ValueError('Target points must be finite')
            theta_route = self._semantic_similarity(candidates, target)
            theta_neighbor = self._semantic_similarity(neighbors, target, valid)
            semantic_weight = (theta_route[:, :, None] + theta_neighbor[:, None, :]) / 2.0
            ranks = proximity.argsort(dim=-1).argsort(dim=-1)
            penalty_weights = torch.where(ranks < 4, weights["rho1"], weights["rho2"])
            # Clearance remains nonconstant outside the collision set, so mu2,
            # rho and semantic alignment affect feasible-candidate supervision.
            risk = torch.exp(-clearance.clamp_min(0) / self.risk_scale)
            penalty = (risk * (1.0 + semantic_weight) * penalty_weights[:, None]).sum(-1)
            reference = reference - weights["mu2"] * penalty
        unsafe = collisions.any(-1)
        if self.objective == 'ratio':
            reference = torch.where(unsafe, reference.clamp_max(0) - self.unsafe_penalty, reference)
        if not torch.isfinite(reference).all():
            raise ValueError('ORRP reference scores overflowed')
        return self.transform_scores(reference)

    def transform_scores(self, raw):
        """Preserve the central scale and compress tails monotonically toward +/-B."""
        knee = self.score_bound / 2.0
        magnitude = raw.abs()
        tail = self.score_bound - knee * knee / magnitude.clamp_min(knee)
        return torch.where(magnitude <= knee, raw, raw.sign() * tail)

    def forward(self, bev, fused_features, target_point, ego_vel,
                expert_waypoints=None, neighbors=None, neighbor_mask=None, neighbor_current=None,
                neighbor_current_mask=None, neighbor_radii=None,
                neighbor_extents=None, neighbor_yaws=None, diagnostics=None):
        batch = bev.shape[0]
        speed = ego_vel.reshape(batch, -1)[:, :1].to(bev)
        target = lbc_target_to_ego(target_point.to(bev))
        pooled = F.adaptive_avg_pool2d(bev, 1).flatten(1)
        context = self.context(torch.cat((fused_features, pooled, target, speed), dim=-1))
        hidden = context[:, None, :] + self.mode_embeddings.weight[None]
        hidden = hidden.reshape(batch * self.num_candidates, self.hidden_size)
        point = bev.new_zeros(batch * self.num_candidates, 2)
        mode_target = target[:, None].expand(-1, self.num_candidates, -1).reshape(-1, 2)
        mode_speed = speed[:, None].expand(-1, self.num_candidates, -1).reshape(-1, 1)
        generated = []
        for _ in range(self.pred_len):
            hidden = self.decoder(torch.cat((point, mode_target, mode_speed), dim=-1), hidden)
            point = point + self.waypoint_delta(hidden)
            generated.append(point)
        candidates = torch.stack(generated, dim=1).reshape(batch, self.num_candidates, self.pred_len, 2)
        candidates = candidates + candidates.new_tensor([-self.lidar_x, 0.0])

        _, _, _, headings = trajectory_kinematics(candidates, speed, self.dt, self.lidar_x)
        time = torch.arange(1, self.pred_len + 1, device=bev.device, dtype=bev.dtype) / self.pred_len
        time = time[None, None, :, None].expand(batch, self.num_candidates, -1, -1)
        trajectory_features = torch.cat((candidates / 32.0, headings.sin()[..., None],
                                         headings.cos()[..., None], time), dim=-1)
        queries = self.trajectory_encoder(trajectory_features).flatten(1, 2)
        grid = self.attention_grid
        bev_tokens = self.bev_projection(F.adaptive_avg_pool2d(bev, (grid, grid))).flatten(2).transpose(1, 2)
        axis = torch.linspace(-1.0, 1.0, grid, device=bev.device, dtype=bev.dtype)
        rows, columns = torch.meshgrid(axis, axis, indexing="ij")
        position = torch.stack((rows, columns), dim=-1).reshape(grid * grid, 2)
        bev_tokens = bev_tokens + self.bev_position(position)[None]
        attended, _ = self.cross_attention(queries, bev_tokens, bev_tokens, need_weights=False)
        interaction = self.attention_norm(queries + attended).reshape(batch, self.num_candidates, self.pred_len, -1).mean(2)
        score_features = torch.cat((interaction, hidden.reshape(batch, self.num_candidates, -1),
                                    context[:, None].expand(-1, self.num_candidates, -1)), dim=-1)
        scores = self.score_head(score_features).squeeze(-1)
        diagnostics = (self.training or expert_waypoints is not None) if diagnostics is None else diagnostics
        need_reference = diagnostics or self.selection == 'analytic'
        with torch.no_grad(), torch.autocast(device_type=bev.device.type, enabled=False):
            geometry = self._collision_geometry(candidates.detach(), neighbors, neighbor_mask,
                neighbor_current, neighbor_current_mask, neighbor_radii, neighbor_extents, neighbor_yaws,
                need_clearance=need_reference)
            reference_scores = (self._reference_impl(candidates, speed, target_point, neighbors,
                neighbor_mask, neighbor_current, neighbor_current_mask, neighbor_radii, neighbor_extents,
                neighbor_yaws, geometry=geometry) if need_reference else None)
        reachable = forward_reachable(candidates.detach().float(), speed.float(), self.config)
        feasible = ~geometry[0].any(-1) & reachable
        if not torch.isfinite(scores).all():
            raise ValueError('Learned ORRP scores must be finite')
        loss_behavior = (F.smooth_l1_loss(scores.float(), reference_scores.float())
                         if diagnostics else scores.new_zeros(()))
        if expert_waypoints is None:
            loss_wp = candidates.sum() * 0.0
        else:
            if expert_waypoints.shape != (batch, self.pred_len, 2):
                raise ValueError("expert_waypoints must have shape (B, pred_len, 2)")
            errors = (candidates - expert_waypoints[:, None].to(candidates)).abs().mean(dim=(-1, -2))
            loss_wp = errors.min(dim=1).values.mean() + self.all_candidate_weight * errors.mean()
        selection_scores = reference_scores if self.selection == "analytic" else scores
        feasible = feasible & torch.isfinite(selection_scores)
        selection_scores = selection_scores.masked_fill(~feasible, -torch.inf)
        best = selection_scores.argmax(dim=1)
        waypoints = candidates[torch.arange(batch, device=bev.device), best]
        emergency_stop = ~feasible.any(-1)
        stopped = candidates.new_tensor([-self.lidar_x, 0.0]).expand_as(waypoints)
        waypoints = torch.where(emergency_stop[:, None, None], stopped, waypoints)
        valid_batch = (~emergency_stop).float()
        agreement = regret = scores.new_zeros(())
        if diagnostics:
            learned_best = scores.masked_fill(~feasible, -torch.inf).argmax(1)
            analytic_best = reference_scores.masked_fill(~feasible, -torch.inf).argmax(1)
            agreement = ((learned_best == analytic_best).float() * valid_batch).sum() / valid_batch.sum().clamp_min(1)
            rows = torch.arange(batch, device=bev.device)
            regret = reference_scores[rows, analytic_best] - reference_scores[rows, learned_best]
            regret = torch.where(emergency_stop, torch.zeros_like(regret), regret).sum() / valid_batch.sum().clamp_min(1)
        selected_indices = best.masked_fill(emergency_stop, -1)
        return {"waypoints": waypoints, "selected_indices": selected_indices, "candidates": candidates, "scores": scores,
                "reference_scores": reference_scores, "loss_wp": loss_wp, "loss_behavior": loss_behavior,
                "emergency_stop": emergency_stop, "feasible": feasible, "kinematically_feasible": reachable,
                "metric_rank_agreement": agreement, "metric_score_regret": regret,
                "metric_rank_valid_fraction": valid_batch.mean(),
                "metric_score_saturation": ((reference_scores.abs() >= .99 * self.score_bound).float().mean()
                    if diagnostics else scores.new_zeros(()))}

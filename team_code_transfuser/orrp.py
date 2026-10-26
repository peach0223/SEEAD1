import math
import warnings

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
        self.grouping_policy = _parameter(config, "bev_orrp_grouping_policy", "legacy_nearest_four")
        self._legacy_grouping_warned = False
        self.collision_radius = float(_parameter(config, "bev_orrp_collision_radius", 2.5))
        self.attention_grid = int(_parameter(config, "bev_orrp_attention_grid", 8))
        self.objective = _parameter(config, 'bev_score_objective', 'ratio')
        self.selection = _parameter(config, "bev_score_selection",
                                    "analytic" if self.objective == "ratio" else "learned")
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
        if self.grouping_policy not in ("explicit", "legacy_nearest_four"):
            raise ValueError("bev_orrp_grouping_policy must be explicit or legacy_nearest_four")
        if min(self.pred_len, self.num_candidates, self.hidden_size, self.attention_grid) < 1:
            raise ValueError("Trajectory and attention dimensions must be positive")
        if min(self.dt, self.cost_floor, self.collision_radius) <= 0:
            raise ValueError("Waypoint dt, control cost floor and collision radius must be positive")
        if self.selection not in ("learned", "analytic"):
            raise ValueError("bev_score_selection must be 'learned' or 'analytic'")
        if self.objective == "ratio" and self.selection == "learned":
            warnings.warn("bev_score_selection='learned' is a compatibility alias for 'analytic' "
                          "in ratio mode: final selection uses raw RR; learned scores remain "
                          "available for behavior MSE and diagnostics.", UserWarning, stacklevel=2)
            self.selection = "analytic"
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
        self.score_head = nn.Sequential(nn.Linear(3 * hidden + 1 + 2 * self.pred_len, hidden),
                                        nn.SiLU(), nn.Linear(hidden, 1))


        self.interval_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.speed_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.heading_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        for head in (self.interval_head, self.speed_head, self.heading_head):
            nn.init.normal_(head[-1].weight, std=0.01)
            nn.init.zeros_(head[-1].bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        physical_keys = [prefix + name for name in self.state_dict()
                         if name.startswith(("interval_head.", "speed_head.", "heading_head."))]
        if any(key not in state_dict for key in physical_keys):
            warnings.warn("ORRP checkpoint is missing physical-head parameters. Strict loading "
                          "will reject it; non-strict loading leaves newly initialized parameters. "
                          "Old scores are not equivalent to the physical-head RR model.",
                          UserWarning, stacklevel=2)
        score_weight = state_dict.get(prefix + "score_head.0.weight")
        if score_weight is not None and score_weight.shape != self.score_head[0].weight.shape:
            warnings.warn("ORRP score_head now consumes time, speed and heading predictions; "
                          "this checkpoint has an incompatible input shape.", UserWarning, stacklevel=2)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def _weights(self):
        return {name: float(_parameter(self.config, "bev_orrp_" + name, 1.0))
                for name in ("eta1", "eta2", "mu1", "mu2", "rho1", "rho2")}

    def _control_cost(self, interval, speeds, heading_changes, ego_vel):

        if (not torch.isfinite(interval).all() or (interval <= 0).any()
                or not torch.isfinite(speeds).all() or (speeds < 0).any()
                or not torch.isfinite(heading_changes).all()
                or not torch.isfinite(ego_vel).all()):
            raise ValueError("ORRP physical quantities must be finite with positive intervals and nonnegative speeds")
        initial = ego_vel.reshape(speeds.shape[0], -1)[:, :1].to(speeds)
        initial = initial[:, None].expand(-1, speeds.shape[1], -1)
        accelerations = torch.diff(torch.cat((initial, speeds), -1), dim=-1) / interval
        weights = self._weights()
        acceleration_cost = weights["eta1"] * accelerations.abs().sum(-1)
        heading_cost = weights["eta2"] * (heading_changes / interval).abs().sum(-1)
        return acceleration_cost, heading_cost

    def _objective_scores(self, cost, reward, penalty):

        if not all(torch.isfinite(value).all() for value in (cost, reward, penalty)):
            raise ValueError("ORRP cost, reward and penalty must be finite")
        driving_reward = reward - penalty
        if self.objective == "cost":
            scores = -cost
        elif self.objective == "reward":
            scores = driving_reward
        else:
            if (cost <= 0).any():
                raise ValueError("Paper RR requires finite positive raw cost; zero cost is undefined. "
                                 "Use reference details for the separate execution protection policy.")
            scores = 1.0 + driving_reward / cost
        if not torch.isfinite(scores).all():
            raise ValueError("ORRP reference scores overflowed")
        return scores

    def _reference_score_state(self, raw_cost, reward, penalty):
        finite_inputs = torch.isfinite(raw_cost) & torch.isfinite(reward) & torch.isfinite(penalty)
        domain_valid = finite_inputs & (raw_cost > 0)
        sanitized_reward = torch.where(finite_inputs, reward, torch.zeros_like(reward))
        sanitized_penalty = torch.where(finite_inputs, penalty, torch.zeros_like(penalty))
        numerator = sanitized_reward - sanitized_penalty
        denominator = torch.where(domain_valid, raw_cost, torch.ones_like(raw_cost))
        paper_scores = 1.0 + numerator / denominator
        paper_valid = domain_valid & torch.isfinite(paper_scores)
        paper_scores = torch.where(paper_valid, paper_scores, torch.zeros_like(paper_scores))
        execution_input_valid = finite_inputs & (raw_cost >= 0)
        execution_cost = torch.where(domain_valid, raw_cost, torch.full_like(raw_cost, self.cost_floor))
        if self.objective == "cost":
            reference = -raw_cost
            reference_valid = finite_inputs & (raw_cost >= 0)
            execution_scores = -execution_cost
        elif self.objective == "reward":
            reference = numerator
            reference_valid = finite_inputs & torch.isfinite(reference)
            execution_scores = numerator
        else:
            reference = paper_scores
            reference_valid = paper_valid
            execution_scores = 1.0 + numerator / execution_cost
        execution_valid = execution_input_valid & torch.isfinite(execution_scores)
        reference = torch.where(reference_valid, reference, torch.zeros_like(reference))
        execution_scores = torch.where(execution_valid, execution_scores, torch.zeros_like(execution_scores))
        return {"raw_cost": raw_cost, "reference_cost": raw_cost, "reference_scores": reference,
                "reference_score_valid_mask": reference_valid, "paper_scores": paper_scores,
                "paper_score_valid_mask": paper_valid, "paper_score_domain_mask": domain_valid,
                "paper_score_placeholder": 0.0,
                "score_invalid_mask": ~paper_valid,
                "score_overflow_mask": domain_valid & ~paper_valid,
                "execution_cost": execution_cost, "execution_scores": execution_scores,
                "execution_score_valid_mask": execution_valid,
                "cost_protection_mask": execution_input_valid & (raw_cost == 0),
                "execution_cost_policy": "zero_cost_floor_fallback_only",
                "configured_cost_floor": self.cost_floor,
                "distance_reward": reward, "collision_penalty": penalty}

    def _predict_physics(self, interaction, candidates, ego_vel):


        time_logits = self.interval_head(interaction.mean(2)).float()
        speed_logits = self.speed_head(interaction).squeeze(-1).float()
        heading_logits = self.heading_head(interaction).squeeze(-1).float()


        with torch.no_grad(), torch.autocast(device_type=candidates.device.type, enabled=False):
            distances, _, rates, _ = trajectory_kinematics(
                candidates.detach().float(), ego_vel.detach().float(), self.dt, self.lidar_x)
            heading_changes = rates * self.dt
        interval = self.dt * (1.0 + 0.5 * time_logits.tanh())
        speeds = distances / interval * torch.exp(0.5 * speed_logits.tanh())
        changes = wrap_angle(heading_changes + (math.pi / 4) * heading_logits.tanh())
        changes = torch.where(distances > 1e-6, changes, torch.zeros_like(changes))
        acceleration_cost, heading_cost = self._control_cost(interval, speeds, changes, ego_vel)
        return {"predicted_interval": interval, "predicted_speeds": speeds,
                "predicted_heading_changes": changes,
                "predicted_acceleration_cost": acceleration_cost,
                "predicted_heading_cost": heading_cost,
                "predicted_cost": acceleration_cost + heading_cost,
                "predicted_raw_cost": acceleration_cost + heading_cost}

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

        moving = motion.abs() > torch.finfo(motion.dtype).eps
        denominator = torch.where(moving, motion, torch.ones_like(motion))
        first, second = (-radii - offset) / denominator, (radii - offset) / denominator
        lower = torch.where(moving, torch.minimum(first, second), -torch.inf)
        upper = torch.where(moving, torch.maximum(first, second), torch.inf)
        separated = (~moving & (offset.abs() > radii)).any(-1)
        return (lower.amax(-1).clamp_min(0) <= upper.amin(-1).clamp_max(1)) & ~separated

    @staticmethod
    def _minimum_clearance(offset, motion, radii):


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

    def _construct_interaction_pairs(self, candidates, neighbors, neighbor_current_mask):
        batch = candidates.shape[0]
        count = 0 if neighbors is None else neighbors.shape[1]
        neighbor_indices = torch.arange(count, device=candidates.device)
        pairs = torch.stack((torch.full_like(neighbor_indices, -1), neighbor_indices), -1)
        pairs = pairs[None].expand(batch, -1, -1)
        if count == 0:
            active = torch.zeros((batch, 0), dtype=torch.bool, device=candidates.device)
        else:
            if neighbor_current_mask is None or neighbor_current_mask.shape != (batch, count):
                raise ValueError("Interaction pairs require the current neighbor mask")
            active = neighbor_current_mask.to(device=candidates.device, dtype=torch.bool)
        return pairs, active

    def _interaction_groups(self, proximity, pair_active, interaction_group_indices):
        if interaction_group_indices is not None:
            groups = torch.as_tensor(interaction_group_indices, device=pair_active.device)
            if groups.shape == pair_active.shape[1:]:
                groups = groups[None].expand_as(pair_active)
            if groups.shape != pair_active.shape:
                raise ValueError("interaction_group_indices must have shape (M,) or (B, M)")
            if not ((groups == -1) | (groups == 1) | (groups == 2)).all():
                raise ValueError("Interaction group indices must be 1 (rho1), 2 (rho2), or -1 (excluded)")
            source = "explicit_indices"
            groups = groups.to(torch.long)
        elif pair_active.shape[1] == 0:
            groups = torch.empty_like(pair_active, dtype=torch.long)
            source = "no_interaction_pairs"
        elif not pair_active.any():
            groups = torch.full_like(pair_active, -1, dtype=torch.long)
            source = "no_active_interaction_pairs"
        elif self.grouping_policy == "explicit":
            raise ValueError("Explicit ORRP grouping requires interaction_group_indices; "
                             "the paper does not uniquely define the rho groups")
        else:
            if not self._legacy_grouping_warned:
                warnings.warn("ORRP uses legacy_nearest_four grouping for ego-other pairs. "
                              "This compatibility assumption is not a uniquely specified paper grouping "
                              "and does not include neighbor-neighbor pairs; provide explicit group indices.",
                              UserWarning, stacklevel=3)
                self._legacy_grouping_warned = True
            ranks = proximity.argsort(dim=-1).argsort(dim=-1)
            groups = torch.where(ranks < 4, 1, 2)
            source = "legacy_nearest_four"
        return groups.masked_fill(~pair_active, -1), source

    def _weighted_interaction_penalty(self, risk, semantic_weight, group_indices, pair_active):
        if risk.shape != semantic_weight.shape or risk.shape[0:1] + risk.shape[2:] != group_indices.shape:
            raise ValueError("Risk, semantic weights and interaction groups must align")
        if not torch.isfinite(risk).all() or not torch.isfinite(semantic_weight).all():
            raise ValueError("Interaction risk and semantic weights must be finite")
        weights = self._weights()
        rho = torch.where(group_indices == 1, weights["rho1"], weights["rho2"])
        rho = torch.where(pair_active & (group_indices != -1), rho, torch.zeros_like(rho))
        contributions = weights["mu2"] * risk * semantic_weight * rho[:, None]
        return contributions.sum(-1), contributions

    @torch.no_grad()
    def compute_reference_scores(self, candidates, ego_vel, target_point=None,
                                 neighbors=None, neighbor_mask=None, neighbor_current=None,
                                 neighbor_current_mask=None, neighbor_radii=None,
                                 neighbor_extents=None, neighbor_yaws=None, *, physical_quantities=None,
                                 interaction_group_indices=None, return_details=False):

        if physical_quantities is None:
            warnings.warn("No physical_quantities supplied: computing legacy geometric RR with fixed dt. "
                          "Pass physical_quantities=plan to reproduce the attention-based reference scores.",
                          UserWarning, stacklevel=2)
        with torch.autocast(device_type=candidates.device.type, enabled=False):
            return self._reference_impl(candidates, ego_vel, target_point, neighbors, neighbor_mask,
                neighbor_current, neighbor_current_mask, neighbor_radii, neighbor_extents, neighbor_yaws,
                physical_quantities=physical_quantities, interaction_group_indices=interaction_group_indices,
                return_details=return_details)

    def _reference_impl(self, candidates, ego_vel, target_point, neighbors, neighbor_mask,
                        neighbor_current, neighbor_current_mask, neighbor_radii, neighbor_extents, neighbor_yaws,
                        geometry=None, return_components=False, physical_quantities=None,
                        interaction_group_indices=None, return_details=False):
        if candidates.ndim != 4 or candidates.shape[-2:] != (self.pred_len, 2):
            raise ValueError("candidates must have shape (B, N, pred_len, 2)")
        dtype = torch.float64 if candidates.dtype == torch.float64 else torch.float32
        candidates = candidates.to(dtype)
        if not torch.isfinite(candidates).all() or not torch.isfinite(ego_vel).all():
            raise ValueError('Candidate trajectories and ego velocity must be finite')
        ego_vel = ego_vel.to(candidates)
        weights = self._weights()
        distances, _, heading_rates, _ = trajectory_kinematics(
            candidates, ego_vel, self.dt, self.lidar_x)
        if physical_quantities is None:
            interval = candidates.new_tensor(self.dt)
            speeds = distances / self.dt
            heading_changes = heading_rates * self.dt
        else:
            quantities = []
            for name, shape in (("predicted_interval", candidates.shape[:2] + (1,)),
                                ("predicted_speeds", candidates.shape[:-1]),
                                ("predicted_heading_changes", candidates.shape[:-1])):
                value = physical_quantities.get(name)
                if not isinstance(value, torch.Tensor) or value.shape != shape:
                    raise ValueError("Invalid physical quantity shape: " + name)
                quantities.append(value.to(candidates))
            interval, speeds, heading_changes = quantities
        acceleration_cost, heading_cost = self._control_cost(interval, speeds, heading_changes, ego_vel)
        cost = acceleration_cost + heading_cost
        reward = weights["mu1"] * distances.sum(-1)
        penalty = torch.zeros_like(reward)
        _, proximity, clearance = geometry if geometry is not None else self._collision_geometry(candidates, neighbors, neighbor_mask,
            neighbor_current, neighbor_current_mask, neighbor_radii, neighbor_extents, neighbor_yaws)
        pairs, pair_active = self._construct_interaction_pairs(candidates, neighbors, neighbor_current_mask)
        groups, group_source = self._interaction_groups(proximity, pair_active, interaction_group_indices)
        risk = torch.exp(-clearance.clamp_min(0) / self.risk_scale)
        semantic_weight = torch.zeros_like(risk)
        contributions = torch.zeros_like(risk)
        if neighbors is not None and neighbors.shape[1] > 0:
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
            penalty, contributions = self._weighted_interaction_penalty(risk, semantic_weight, groups, pair_active)
        if return_details:
            return {**self._reference_score_state(cost, reward, penalty),
                    "interaction_pairs": pairs, "interaction_pair_active_mask": pair_active,
                    "interaction_pair_scope": "candidate_ego_to_neighbor_only",
                    "interaction_ego_index": -1, "includes_neighbor_neighbor_pairs": False,
                    "interaction_group_indices": groups, "interaction_group_source": group_source,
                    "interaction_risk": risk, "interaction_semantic_weight": semantic_weight,
                    "interaction_penalty_contributions": contributions,
                    "interaction_risk_function": "exp(-max(swept_rectangle_clearance, 0) / risk_scale)",
                    "reference_physics_source": ("supplied_physical_quantities" if physical_quantities is not None
                                                  else "legacy_geometry_fixed_dt")}
        reference = self._objective_scores(cost, reward, penalty)
        if return_components:
            return reference, cost, reward, penalty
        return reference

    def transform_scores(self, raw):

        knee = self.score_bound / 2.0
        magnitude = raw.abs()
        tail = self.score_bound - knee * knee / magnitude.clamp_min(knee)
        return torch.where(magnitude <= knee, raw, raw.sign() * tail)

    def rank_paper_candidates(self, candidates, paper_scores, paper_score_valid_mask=None):
        if paper_scores.shape != candidates.shape[:2]:
            raise ValueError("Paper scores must match candidate batch and count")
        finite = torch.isfinite(paper_scores)
        if paper_score_valid_mask is None:
            if not finite.all():
                raise ValueError("Nonfinite paper scores require an explicit validity mask")
            valid = finite
        else:
            if paper_score_valid_mask.shape != paper_scores.shape:
                raise ValueError("Paper score validity mask must match scores")
            valid = paper_score_valid_mask.to(device=paper_scores.device, dtype=torch.bool) & finite
        valid = valid & torch.isfinite(candidates).all(dim=(-1, -2))
        available = valid.any(-1)
        best = paper_scores.masked_fill(~valid, -torch.inf).argmax(-1)
        rows = torch.arange(candidates.shape[0], device=candidates.device)
        waypoints = candidates[rows, best]
        placeholder = candidates.new_tensor([-self.lidar_x, 0.0]).expand_as(waypoints)
        waypoints = torch.where(available[:, None, None], waypoints, placeholder)
        return {"paper_best_index": best.masked_fill(~available, -1),
                "paper_waypoints": waypoints, "paper_selection_available": available,
                "paper_waypoints_placeholder": "ego_origin_when_no_defined_rr",
                "paper_selection_source": "raw_rr_argmax_without_execution_filters"}

    def apply_execution_protection(self, candidates, selection_scores, feasible,
                                   paper_selection, score_valid_mask=None, cost_protection_mask=None):
        if selection_scores.shape != candidates.shape[:2] or feasible.shape != selection_scores.shape:
            raise ValueError("Execution scores and feasibility must match candidates")
        valid = torch.isfinite(selection_scores) & torch.isfinite(candidates).all(dim=(-1, -2))
        if score_valid_mask is not None:
            if score_valid_mask.shape != valid.shape:
                raise ValueError("Execution score validity mask must match scores")
            valid = valid & score_valid_mask
        feasible = feasible.to(dtype=torch.bool) & valid
        best = selection_scores.masked_fill(~feasible, -torch.inf).argmax(-1)
        rows = torch.arange(candidates.shape[0], device=candidates.device)
        waypoints = candidates[rows, best]
        emergency_stop = ~feasible.any(-1)
        stopped = candidates.new_tensor([-self.lidar_x, 0.0]).expand_as(waypoints)
        waypoints = torch.where(emergency_stop[:, None, None], stopped, waypoints)
        safe_best = best.masked_fill(emergency_stop, -1)
        changed = emergency_stop | (safe_best != paper_selection["paper_best_index"])
        numerical_protection = torch.zeros_like(emergency_stop)
        if cost_protection_mask is not None:
            if cost_protection_mask.shape != feasible.shape:
                raise ValueError("Cost protection mask must match scores")
            numerical_protection = cost_protection_mask.any(-1)
        return {"safe_best_index": safe_best, "selected_indices": safe_best,
                "waypoints": waypoints, "safe_waypoints": waypoints,
                "emergency_stop": emergency_stop, "feasible": feasible,
                "protection_changed_selection": changed,
                "protection_triggered": changed | (~feasible).any(-1) | numerical_protection,
                "execution_selection_source": "collision_reachability_and_numeric_protection"}

    def forward(self, bev, fused_features, target_point, ego_vel,
                expert_waypoints=None, neighbors=None, neighbor_mask=None, neighbor_current=None,
                neighbor_current_mask=None, neighbor_radii=None,
                neighbor_extents=None, neighbor_yaws=None, diagnostics=None, interaction_group_indices=None):
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
        interaction = self.attention_norm(queries + attended).reshape(batch, self.num_candidates, self.pred_len, -1)
        physics = self._predict_physics(interaction, candidates, speed)
        physical_features = torch.cat((physics["predicted_interval"] / self.dt,
                                       physics["predicted_speeds"] / 32.0,
                                       physics["predicted_heading_changes"] / math.pi), dim=-1).to(interaction)
        score_features = torch.cat((interaction.mean(2), hidden.reshape(batch, self.num_candidates, -1),
                                    context[:, None].expand(-1, self.num_candidates, -1), physical_features), dim=-1)
        scores = self.score_head(score_features).squeeze(-1).float()
        diagnostics = (self.training or expert_waypoints is not None) if diagnostics is None else diagnostics


        with torch.no_grad(), torch.autocast(device_type=bev.device.type, enabled=False):
            geometry = self._collision_geometry(candidates.detach(), neighbors, neighbor_mask,
                neighbor_current, neighbor_current_mask, neighbor_radii, neighbor_extents, neighbor_yaws,
                need_clearance=True)
        with torch.autocast(device_type=bev.device.type, enabled=False):
            reference_details = self._reference_impl(candidates.detach(), speed, target_point, neighbors,
                neighbor_mask, neighbor_current, neighbor_current_mask, neighbor_radii, neighbor_extents,
                neighbor_yaws, geometry=geometry, physical_quantities=physics,
                interaction_group_indices=interaction_group_indices, return_details=True)
        physics_scores = reference_details["reference_scores"]
        reference_scores = physics_scores.detach()
        reference_details = {key: value.detach() if isinstance(value, torch.Tensor) else value
                             for key, value in reference_details.items()}
        reference_details["reference_physics_source"] = "attention_physical_heads"
        score_residual = scores - reference_scores


        reachable = forward_reachable(candidates.detach().float(), speed.float(), self.config)
        feasible = ~geometry[0].any(-1) & reachable
        if not torch.isfinite(scores).all():
            raise ValueError('Learned ORRP scores must be finite')
        loss_behavior = (F.mse_loss(scores.float(), reference_scores.float(), reduction="mean")
                         if diagnostics else scores.new_zeros(()))
        if expert_waypoints is None:
            loss_wp = candidates.sum() * 0.0
        else:
            if expert_waypoints.shape != (batch, self.pred_len, 2):
                raise ValueError("expert_waypoints must have shape (B, pred_len, 2)")
            errors = (candidates - expert_waypoints[:, None].to(candidates)).abs().mean(dim=(-1, -2))
            loss_wp = errors.min(dim=1).values.mean() + self.all_candidate_weight * errors.mean()
        paper_selection = self.rank_paper_candidates(candidates, reference_details["paper_scores"],
                                                     reference_details["paper_score_valid_mask"])
        analytic_selection = self.objective == "ratio" or self.selection == "analytic"
        selection_scores = reference_details["execution_scores"] if analytic_selection else scores
        execution = self.apply_execution_protection(candidates, selection_scores, feasible, paper_selection,
                    score_valid_mask=reference_details["execution_score_valid_mask"],
                    cost_protection_mask=reference_details["cost_protection_mask"])
        execution["execution_score_source"] = ("reference_rr_with_explicit_zero_cost_fallback"
                                               if self.objective == "ratio" else
                                               "reference_objective" if analytic_selection else "learned_scores")
        feasible, emergency_stop = execution["feasible"], execution["emergency_stop"]
        diagnostic_feasible = feasible & reference_details["reference_score_valid_mask"]
        diagnostic_available = diagnostic_feasible.any(-1)
        valid_batch = diagnostic_available.float()
        agreement = regret = scores.new_zeros(())
        if diagnostics:
            learned_best = scores.masked_fill(~diagnostic_feasible, -torch.inf).argmax(1)
            analytic_best = reference_scores.masked_fill(~diagnostic_feasible, -torch.inf).argmax(1)
            agreement = ((learned_best == analytic_best).float() * valid_batch).sum() / valid_batch.sum().clamp_min(1)
            rows = torch.arange(batch, device=bev.device)
            regret = reference_scores[rows, analytic_best] - reference_scores[rows, learned_best]
            regret = torch.where(diagnostic_available, regret, torch.zeros_like(regret)).sum() / valid_batch.sum().clamp_min(1)
        return {**physics, **reference_details, **paper_selection, **execution,
                "physics_scores": physics_scores, "score_residual": score_residual,
                "candidates": candidates, "scores": scores,
                "reference_scores": reference_scores, "loss_wp": loss_wp, "loss_behavior": loss_behavior,
                "emergency_stop": emergency_stop, "feasible": feasible, "kinematically_feasible": reachable,
                "metric_rank_agreement": agreement, "metric_score_regret": regret,
                "metric_rank_valid_fraction": valid_batch.mean(),


                "metric_score_saturation": ((reference_scores.abs() >= .99 * self.score_bound).float().mean()
                    if diagnostics else scores.new_zeros(()))}

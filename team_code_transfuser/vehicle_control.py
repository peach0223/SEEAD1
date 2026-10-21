"""Forward-only PID execution and the repository's fitted bicycle model."""
import math
from collections.abc import Mapping
from copy import deepcopy

import numpy as np


def validate_control_timing(config):
    """Keep sensor time, waypoint time and the control update cadence consistent."""
    get = config.get if isinstance(config, Mapping) else lambda name, default: getattr(config, name, default)
    frame_dt = get('carla_frame_rate', .05)
    waypoint_dt = get('bev_waypoint_dt', .5)
    repeat = get('action_repeat', 2)
    for name, value in (('carla_frame_rate', frame_dt), ('bev_waypoint_dt', waypoint_dt)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(name + ' must be finite and positive')
    if type(repeat) is not int or repeat < 1:
        raise ValueError('action_repeat must be a positive integer')
    fps = get('carla_fps', 1. / frame_dt)
    if (isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps)
            or not math.isclose(fps * frame_dt, 1., rel_tol=1e-7, abs_tol=1e-9)):
        raise ValueError('carla_fps must agree with carla_frame_rate')
    if get('bev_enabled', False) and get('bev_history_frames', 0):
        stride = waypoint_dt / (frame_dt * repeat)
        if stride < 1 or not math.isclose(stride, round(stride), rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError('BEV history interval must be a whole number of control updates')
    return float(frame_dt), float(waypoint_dt), repeat


class EgoModel:
    # Existing World on Rails fit, shared by GPS prediction and control checks.
    front_wb = -0.090769015
    rear_wb = 1.4178275
    steer_gain = 0.36848336
    brake_accel = -4.952399
    throt_accel = 0.5633837

    def __init__(self, dt=.25):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('Vehicle integration interval must be positive')
        self.dt = dt

    def forward(self, locs, yaws, spds, acts):
        steer, throttle, brake = np.asarray(acts).reshape(3)
        yaw, speed = float(np.asarray(yaws).item()), float(np.asarray(spds).item())
        acceleration = self.brake_accel if brake else self.throt_accel * throttle
        beta = math.atan(self.rear_wb / (self.front_wb + self.rear_wb) * math.tan(self.steer_gain * steer))
        direction = np.array([math.cos(yaw + beta), math.sin(yaw + beta)])
        return (np.asarray(locs) + speed * direction * self.dt,
                np.array(yaw + speed / self.rear_wb * math.sin(beta) * self.dt),
                np.array(max(0., speed + acceleration * self.dt)))


def forward_reachable(waypoints, speed, config):
    """Necessary longitudinal/steering bounds; execution is checked separately."""
    import torch
    frame_dt, dt, _ = validate_control_timing(config)
    origin = waypoints.new_tensor([-getattr(config, 'lidar_pos', [1.3])[0], 0.])
    steps = torch.diff(torch.cat((origin.expand(*waypoints.shape[:-2], 1, 2), waypoints), -2), dim=-2)
    distance = torch.linalg.vector_norm(steps, dim=-1)
    velocity = distance / dt
    initial = speed.reshape(waypoints.shape[0], -1)[:, :1].to(waypoints)
    initial = initial[:, None].expand(-1, waypoints.shape[1], -1)
    acceleration = torch.diff(torch.cat((initial, velocity), -1), dim=-1) / dt
    max_acceleration = EgoModel.throt_accel * float(getattr(config, 'clip_throttle', .75))
    longitudinal = ((acceleration >= EgoModel.brake_accel - 1e-4) &
                    (acceleration <= max_acceleration + 1e-4)).all(-1)
    # The first interval starts at the measured speed, not its segment average.
    step = min(frame_dt, dt)
    minimum = torch.zeros_like(initial[..., 0])
    maximum = torch.zeros_like(minimum)
    elapsed = 0.
    while elapsed < dt - 1e-9:
        duration = min(step, dt - elapsed)
        minimum += (initial[..., 0] + EgoModel.brake_accel * elapsed).clamp_min(0) * duration
        maximum += (initial[..., 0] + max_acceleration * elapsed) * duration
        elapsed += duration
    beta = math.atan(EgoModel.rear_wb / (EgoModel.front_wb + EgoModel.rear_wb) * math.tan(EgoModel.steer_gain))
    curvature = math.sin(beta) / EgoModel.rear_wb
    # A bounded-curvature arc can be shorter in chord distance than arc length.
    half_turn = (maximum * curvature / 2).clamp_max(math.pi)
    chord_factor = torch.sinc(half_turn / math.pi).clamp_min(0)
    first_interval = ((distance[..., 0] >= minimum * chord_factor - 1e-4) &
                      (distance[..., 0] <= maximum + 1e-4))
    moving = distance > 1e-5
    headings = torch.atan2(torch.where(moving, steps[..., 1], torch.zeros_like(distance)),
                           torch.where(moving, steps[..., 0], torch.ones_like(distance)))
    previous = torch.zeros_like(headings[..., 0])
    steering = torch.ones_like(longitudinal)
    for index in range(waypoints.shape[-2]):
        heading = torch.where(moving[..., index], headings[..., index], previous)
        difference = torch.atan2(torch.sin(heading - previous), torch.cos(heading - previous)).abs()
        # Slip angle may change with steering; yaw change is bounded by arc length.
        allowance = velocity[..., index] * dt * curvature + beta * (1 if index == 0 else 2)
        steering &= difference <= allowance + 1e-4
        previous = heading
    return (longitudinal & first_interval & steering & torch.isfinite(waypoints).all(dim=(-1, -2))
            & torch.isfinite(initial[..., 0]) & (initial[..., 0] >= 0) & (steps[..., 0, 0] >= -1e-4))


def pid_command(waypoints, speed, config, turn_controller, speed_controller):
    points = np.asarray(waypoints, dtype=np.float64).copy()
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('PID control requires finite waypoints')
    if not math.isfinite(speed) or speed < 0:
        raise ValueError('PID control requires a nonnegative finite speed')
    points[:, 0] += config.lidar_pos[0]
    if points[0, 0] < -1e-4:
        return 0., 0., True
    desired_speed = np.linalg.norm(points[0]) / getattr(config, 'bev_waypoint_dt', .5)
    # A small positive displacement can be an executable low-speed start.
    if desired_speed < 1e-4:
        return 0., 0., True
    minimum_speed = 1e-4 if getattr(config, "bev_orrp_enabled", False) else config.brake_speed
    brake = desired_speed < minimum_speed or speed / max(desired_speed, 1e-6) > config.brake_ratio
    delta = np.clip(desired_speed - speed, 0., config.clip_delta)
    throttle = np.clip(speed_controller.step(delta), 0., config.clip_throttle)
    aim = (points[0] + points[1]) / 2.
    angle = math.atan2(aim[1], aim[0]) / (math.pi / 2)
    if speed < .01 or brake:
        angle = 0.
    steer = np.clip(turn_controller.step(angle), -1., 1.)
    return float(steer), float(0. if brake else throttle), bool(brake)


def rollout_control(waypoints, speed, config, turn_controller, speed_controller, braking=False):
    """Execute the same PID at action cadence on a fixed, timestamped reference.

    Returned positions and body yaws use the virtual LiDAR coordinate frame.
    This is a prediction under the fitted model, not a CARLA safety guarantee.
    """
    turn_controller, speed_controller = deepcopy(turn_controller), deepcopy(speed_controller)
    frame_dt, dt, repeat = validate_control_timing(config)
    path = np.asarray(waypoints, dtype=np.float64).copy()
    path[:, 0] += config.lidar_pos[0]
    path = np.vstack((np.zeros(2), path))
    times = np.arange(len(path)) * dt
    horizon = max(times[-1], speed / -EgoModel.brake_accel + frame_dt) if braking else times[-1]
    count = int(math.ceil(horizon / frame_dt))
    vehicle = EgoModel(frame_dt)
    position, yaw, velocity = np.zeros(2), np.array(0.), np.array(speed)
    positions, yaws, velocities, controls = [], [], [], []
    control = (0., 0., True)
    for index in range(count):
        if index % repeat == 0:
            segment = min(int(index * frame_dt / dt), len(path) - 2)
            planned_hold = np.linalg.norm(path[segment + 1] - path[segment]) < 1e-5
            if braking or planned_hold:
                control = (0., 0., True)
            else:
                queries = index * frame_dt + np.array([dt, 2 * dt])
                future = np.stack([np.interp(queries, times, path[:, axis]) for axis in range(2)], axis=-1)
                # A finite prediction horizon does not command a stop at its endpoint.
                # Continue its terminal velocity for PID lookahead only; collision
                # validation still covers the full requested rollout horizon.
                terminal_velocity = (path[-1] - path[-2]) / dt
                future += np.maximum(queries - times[-1], 0.)[:, None] * terminal_velocity
                c, s = math.cos(float(yaw)), math.sin(float(yaw))
                local = (future - position) @ np.array([[c, -s], [s, c]])
                local[:, 0] -= config.lidar_pos[0]
                control = pid_command(local, float(velocity), config, turn_controller, speed_controller)
        position, yaw, velocity = vehicle.forward(position, yaw, velocity, np.asarray(control))
        positions.append(position.copy() - np.array([config.lidar_pos[0], 0.]))
        yaws.append(float(yaw)); velocities.append(float(velocity)); controls.append(control)
    return {'positions': np.asarray(positions), 'headings': np.asarray(yaws),
            'speeds': np.asarray(velocities), 'controls': np.asarray(controls),
            'times': np.arange(1, count + 1) * frame_dt}

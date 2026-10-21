# SEEAD

Selected model components for:

**Structuring End-to-End Autonomous Driving: A BEV-Centric Framework with Diffusion-Based Feature Enhancement and Optimal Response Ratio Planning**

This repository provides the BEV feature enhancement and Optimal Response Ratio Planning (ORRP) implementations, together with the underlying and alternative sensor-fusion backbones.

**Release scope.** This release includes the original BEV enhancement and ORRP source modules. The BEV enhancement implementation covers VIB, STGSM, layout conditioning, and the conditional diffusion network. Configuration files, training and evaluation pipelines, datasets, and trained weights are not included. The complete driving-model integration remains outside this component release.

## Model components

### BEV feature enhancement

[`team_code_transfuser/bev_enhancement.py`](team_code_transfuser/bev_enhancement.py) contains the original implementations of:

| Component | Function |
| --- | --- |
| `VariationalInformationBottleneck` | Encodes BEV features into a latent representation and computes the KL regularization term. Auxiliary heads provide semantic and waypoint supervision. |
| `SpatioTemporalGatedModulation` | Uses a conditioning vector, spatial attention, and temporal attention to modulate BEV features. |
| `ObjectLayoutEncoder` | Encodes object-layout information into tokens. |
| `LayoutCrossAttention` | Incorporates layout tokens into spatial feature representations. |
| `ConditionalDiffusionUNet` | Predicts noise under semantic, temporal, and layout conditioning. |
| `BEVEnhancement` | Integrates the feature enhancement path and auxiliary diffusion objective, with separate representation, planning, and joint stages. |

The enhancement implementation uses auxiliary diffusion during feature learning and a clean feature path for online enhancement. The VIB waypoint output is an auxiliary supervision output; it is not the final trajectory selected by ORRP.

`VariationalInformationBottleneck` and `SpatioTemporalGatedModulation` take their dimensions directly through constructor arguments. `BEVEnhancement` requires an application-specific configuration object, BEV features, object-layout inputs, and optional historical features. Configuration objects are supplied by the caller.

### Optimal Response Ratio Planning

[`team_code_transfuser/orrp.py`](team_code_transfuser/orrp.py) contains `ORRPPlanner` and its trajectory-processing utilities. The implementation generates candidate waypoint sequences, scores them using BEV and planning features, evaluates trajectory costs and benefits, and selects a trajectory subject to motion and collision checks. It also includes utilities for constructing neighboring-agent information from detection outputs.

The planner imports `forward_reachable` from [`team_code_transfuser/vehicle_control.py`](team_code_transfuser/vehicle_control.py). This dependency is included unchanged and provides motion-feasibility checks, vehicle-motion prediction, and control helpers.

`ORRPPlanner` takes a configuration object and feature dimensions at construction. Its forward interface accepts BEV features, fused features, a navigation target, and ego speed, with optional neighboring-agent inputs and expert waypoints for training.

### Sensor-fusion backbones

| Original file | Contents |
| --- | --- |
| [`transfuser.py`](team_code_transfuser/transfuser.py) | Image and LiDAR encoders, multi-scale transformer fusion, and auxiliary decoders. |
| [`geometric_fusion.py`](team_code_transfuser/geometric_fusion.py) | An alternative geometric fusion backbone. |
| [`late_fusion.py`](team_code_transfuser/late_fusion.py) | An alternative late-fusion backbone. |
| [`latentTF.py`](team_code_transfuser/latentTF.py) | An alternative latent-transformer backbone. |
| [`point_pillar.py`](team_code_transfuser/point_pillar.py) | An optional point-cloud encoder. |

These backbone files are inherited from the source project's TransFuser foundation. Their inclusion does not imply that every alternative was used in the reported experiments. The PointPillar file retains its original credit to Tianwei Yin and the LAV project.


## Acknowledgment

This project benefited from the open-source resources provided by [CARLA](https://carla.org/) and [carla_garage](https://github.com/autonomousvision/carla_garage). Our baseline is developed based on `carla_garage` with further modifications and extensions for this work.

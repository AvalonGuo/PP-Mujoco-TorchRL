from dm_control import mujoco,mjcf
from torchrl.envs import EnvBase
from torchrl.data import Bounded
from ml_collections import config_dict
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
_ARM_JOINTS = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]
_FINGER_JOINTS = ["finger_joint1", "finger_joint2"]
def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt = 0.02,
        sim_dt = 0.005,
        episode_length = 150,
        action_repeat = 1,
        action_scale = 0.04,
    )
class PandaBase(EnvBase):
    def __init__(
        self,
        device = "cpu",
        xml_path = None,
        config:config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
        ):
        super().__init__(device=device)
        if xml_path is None:
            raise ValueError("xml_path can't be None!")
        self._physics = mjcf.Physics.from_xml_path(xml_path)
        self._mj_model = self._physics.model.ptr
        self._mj_data = self._physics.data.ptr
        self.config = config.lock()
        if config_overrides:
           self.config.update_from_flattened_dict(config_overrides)
        self._ctrl_dt = config.ctrl_dt
        self._mj_model.opt.timestep = config.sim_dt

    def _post_init(self, obj_name: str, keyframe: str):
        all_joints = _ARM_JOINTS + _FINGER_JOINTS
        self._robot_arm_qposadr = np.array([
            self._mj_model.jnt_qposadr[self._mj_model.joint(j).id]
            for j in _ARM_JOINTS
        ])
        self._robot_qposadr = np.array([
            self._mj_model.jnt_qposadr[self._mj_model.joint(j).id]
            for j in all_joints
        ])
        self._gripper_site = self._mj_model.site("gripper").id
        self._left_finger_geom = self._mj_model.geom("left_finger_pad").id
        self._right_finger_geom = self._mj_model.geom("right_finger_pad").id
        self._hand_geom = self._mj_model.geom("hand_capsule").id
        self._obj_body = self._mj_model.body(obj_name).id
        self._obj_qposadr = self._mj_model.jnt_qposadr[
            self._mj_model.body(obj_name).jntadr[0]
        ]
        self._mocap_target = self._mj_model.body("mocap_target").mocapid
        self._floor_geom = self._mj_model.geom("floor").id
        self._init_q = self._mj_model.keyframe(keyframe).qpos
        self._init_obj_pos = np.array(
            self._init_q[self._obj_qposadr : self._obj_qposadr + 3],
            dtype=np.float32,
        )
        self._init_ctrl = self._mj_model.keyframe(keyframe).ctrl
        self._lowers, self._uppers = self._mj_model.actuator_ctrlrange.T

    def _reset(self, tensordict, **kwargs):
        return super()._reset(tensordict, **kwargs)
    
    def _set_seed(self, seed):
        return super()._set_seed(seed)
    
    def _step(self, tensordict):
        return super()._step(tensordict)
    
    @property
    def action_spec(self):
        dm_specs = mujoco.action_spec(self._physics)
        th_specs = Bounded(
            low=dm_specs.minimum,
            high=dm_specs.maximum,
            shape=dm_specs.shape,
            dtype=torch.float32
        )
        return th_specs
    
    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model
    

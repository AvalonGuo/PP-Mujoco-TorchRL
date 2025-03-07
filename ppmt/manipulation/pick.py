from ppmt.manipulation.base import PandaBase
from ppmt.manipulation import collision
from ml_collections import config_dict
from typing import Any, Dict, Optional, Union
from tensordict import TensorDict,TensorDictBase
from torchrl.data import Bounded,Unbounded,Composite
from mujoco.mjx._src import math
import mujoco.viewer
import torch
import time
import numpy as np

def gen_info(config,target_pos=None):
  if target_pos is None:
    target_pos = np.zeros((3,))
  info = TensorDict({
    "target_pos":target_pos,
    "reached_box":0.0,
    "out_of_bounds":np.array(0.0,dtype=float),
    **{k: 0.0 for k in config.reward_config.scales.keys()},
  })
  return info

def make_composite_from_td(td):
  composite = Composite(
    {
      key: make_composite_from_td(tensor)
      if isinstance(tensor, TensorDictBase)
      else Unbounded(dtype=tensor.dtype, device=tensor.device, shape=tensor.shape)
      for key, tensor in td.items()
    },
    shape=td.shape
  )
  return composite  

def default_config() -> config_dict.ConfigDict:
  """Returns the default config for bring_to_target tasks."""
  config = config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=150,
      action_repeat=1,
      action_scale=0.04,
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Gripper goes to the box.
              gripper_box=4.0,
              # Box goes to the target mocap.
              box_target=8.0,
              # Do not collide the gripper with the floor.
              no_floor_collision=0.25,
              # Arm stays close to target pose.
              robot_target_qpos=0.3,
          )
      ),
  )
  return config

class PickCubeOrientation(PandaBase):
  def __init__(
    self, 
    device="cpu", 
    xml_path:str="ppmt/manipulation/franka_panda/mjx_single_cube.xml",
    seed:int = 1,
    render_mode:str = "rgb",
    sample_orientation: bool = False, 
    config:config_dict.ConfigDict = default_config(), 
    config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]]  = None):
    super().__init__(device, xml_path, config, config_overrides)

    self._set_seed(seed)
    self._post_init(obj_name="box", keyframe="home")
    self._sample_orientation = sample_orientation
    self._action_scale = config.action_scale
    self.viewer = None
    self.render_mode = render_mode
    self._timestep = self._mj_model.opt.timestep
    info = gen_info(self.config)
    self._make_spec(info)

  def _set_seed(self, seed:Optional[int]):
    rng = torch.Generator()
    if seed is not None:
      rng.manual_seed(seed)
    self.rng = rng

  def _reset(self, tensordict=None):
    objs = self.init_objs_randomly()
    with self._physics.reset_context():
      self._mj_data.qpos = objs["init_q"]
      self._mj_data.qvel = np.zeros(self._mj_model.nv,dtype=float)
      self._mj_data.ctrl = self._init_ctrl
      self._mj_data.mocap_pos[self._mocap_target] = objs["target_pos"]
      self._mj_data.mocap_quat[self._mocap_target] = objs["target_quat"]
    info = gen_info(self.config,objs["target_pos"])
    observation = self._get_obs(info)
    reward = np.zeros(1,dtype=np.float32)
    action = self.action_spec.rand()
    out = TensorDict(
      {
        "action":action,
        "observation":observation,
        "reward":reward,
        "info":info
      }
    )
    return out
  
  def init_objs_randomly(self):
    # intialize box position
    box_pos_minval=np.array([-0.2,-0.2,0.0])
    box_pos_maxval=np.array([0.2, 0.2, 0.0])
    box_pos = (torch.rand(size=(3,),generator=self.rng)*(box_pos_maxval-box_pos_minval)+box_pos_minval).detach().numpy()

    # initialize target position
    target_pos_minval = np.array([-0.2, -0.2, 0.2])
    target_pos_maxval = np.array([0.2, 0.2, 0.4])
    target_pos = (torch.rand(size=(3,),generator=self.rng)*(target_pos_maxval-target_pos_minval)+target_pos_minval).detach().numpy()

    # initialize target orientation
    target_quat = np.array([1.0, 0.0, 0.0, 0.0],dtype=float)
    if self._sample_orientation:
      # sample a random direction
      perturb_axis = torch.rand(size=(3,),generator=self.rng)*2-1
      perturb_axis = perturb_axis / math.norm(perturb_axis)
      perturb_theta = torch.rand(generator=self.rng)*np.deg2rad(45)
      target_quat = math.axis_angle_to_quat(perturb_axis,perturb_theta)

    init_q = np.array(self._init_q)
    init_q[self._obj_qposadr:self._obj_qposadr+3] = box_pos

    return {
      "box_pos":box_pos,
      "target_pos":target_pos,
      "target_quat":target_quat,
      "init_q":init_q
    }
  
  def _get_obs(self,info:TensorDict[str,Any]):
    gripper_pos = self._mj_data.site_xpos[self._gripper_site]
    gripper_mat = self._mj_data.site_xmat[self._gripper_site].ravel()
    target_mat = math.quat_to_mat(self._mj_data.mocap_quat[self._mocap_target])
    observation = np.concatenate([
      self._mj_data.qpos,
      self._mj_data.qvel,
      gripper_pos,
      gripper_mat[3:],
      self._mj_data.xmat[self._obj_body].ravel()[3:],
      self._mj_data.xpos[self._obj_body] - self._mj_data.site_xpos[self._gripper_site],
      info["target_pos"] - self._mj_data.xpos[self._obj_body],
      target_mat.ravel()[:6] - self._mj_data.xmat[self._obj_body].ravel()[:6],
      self._mj_data.ctrl - self._mj_data.qpos[self._robot_qposadr[:-1]],
    ],dtype=np.float32)
    return observation
  
  def _get_rewards(self,info:dict[str,Any]):
    target_pos = info["target_pos"]
    box_pos = self._mj_data.xpos[self._obj_body]
    gripper_pos = self._mj_data.site_xpos[self._gripper_site]
    pos_err = np.linalg.norm(target_pos-box_pos)
    box_mat = self._mj_data.xmat[self._obj_body]
    target_mat = math.quat_to_mat(self._mj_data.mocap_quat[self._mocap_target])
    rot_err = np.linalg.norm(target_mat.ravel()[:6] - box_mat.ravel()[:6])
    box_target = 1 - np.tanh(5 * (0.9 * pos_err + 0.1 * rot_err))
    gripper_box = 1 - np.tanh(5 * np.linalg.norm(box_pos - gripper_pos))
    gripper_box = gripper_box.astype(np.float32)
    robot_target_qpos = 1 - np.tanh(
      np.linalg.norm(
        self._mj_data.qpos[self._robot_arm_qposadr]
        - self._init_q[self._robot_arm_qposadr]
        )
    )
    robot_target_qpos = robot_target_qpos.astype(np.float32)
    hand_floor_collision = [
      collision.geoms_colliding(self._mj_data,self._floor_geom,g)
      for g in [
        self._left_finger_geom,
        self._right_finger_geom,
        self._hand_geom,
        ]
    ]
    floor_collision = sum(hand_floor_collision) > 0
    no_floor_collision = np.array((1.0 - floor_collision),dtype=np.float32)

    info["reached_box"] = 1.0 * np.maximum(
      info["reached_box"],
      (np.linalg.norm(box_pos - gripper_pos) < 0.012),
    )

    rewards = {
      "gripper_box": gripper_box,
      "box_target": box_target * info["reached_box"],
      "no_floor_collision": no_floor_collision,
      "robot_target_qpos": robot_target_qpos,
    }
    return rewards
  
  def _make_spec(self,info):
    self.observation_spec = Composite(
      action = Bounded(
        low=self.action_spec.low,
        high=self.action_spec.high,
        shape=self.action_spec.shape,
        dtype=self.action_spec.dtype
      ),
      observation = Unbounded(shape=(66,)),
      info = make_composite_from_td(info),
      reward = Unbounded(shape=(1,))
    )
    # self.reward_spec = Unbounded(shape=(1,))

  def _step(self, tensordict):
    if self.viewer == None and self.render_mode=="human":
      self.viewer = mujoco.viewer.launch_passive(model=self._mj_model,data=self._mj_data,show_left_ui=True,show_right_ui=True)
    info = tensordict["info"]
    delta = (tensordict["action"]*self._action_scale).detach().numpy()
    ctrl = np.clip(self._mj_data.ctrl + delta,self._lowers,self._uppers)
    ctrl = ctrl.astype(np.float32)
    self._mj_data.ctrl[:] = ctrl[:]
    self._physics.step()
    if self.viewer != None:
      step_start = time.time()
      self.viewer.sync()
      time_until_next_step = self._timestep - (time.time() - step_start)
      if time_until_next_step > 0:
        time.sleep(time_until_next_step)
    raw_rewards = self._get_rewards(info)
    rewards = {
      k: v * self.config.reward_config.scales[k]
      for k,v in raw_rewards.items()
    }
    reward = np.clip(sum(rewards.values()), -1e4, 1e4)
    reward = np.array([reward],dtype=np.float32)
    box_pos = self._mj_data.xpos[self._obj_body]
    out_of_bounds = np.any(np.abs(box_pos) > 1.0)
    out_of_bounds |= box_pos[2] < 0.0
    done = out_of_bounds | np.isnan(self._mj_data.qpos).any() | np.isnan(self._mj_data.qvel).any()
    info["out_of_bounds"] = out_of_bounds.astype(float)
    info.update(raw_rewards)
    observation = self._get_obs(info)
    out = TensorDict({
      "action":ctrl,
      "observation":observation,
      "reward":reward,
      "done":done,
      "info":info},
      batch_size=tensordict.shape
    )
    return out
  
# env = PickCubeOrientation()
# check_env_specs(env)
# td = env._reset()
# for key in td.keys():
#     print("Reset:",key,type(td[key]))
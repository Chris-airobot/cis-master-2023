# Creating object definitions
from robosuite.models import MujocoWorldBase
from robosuite.models.robots import Panda
from robosuite.models.grippers import gripper_factory
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject
from robosuite.utils.mjcf_utils import new_joint
import mujoco

class CustomExample:
    def __init__(self):
        self.world = MujocoWorldBase()
        self.initialize()

    def initialize(self):
        # Creating the robot
        self.mujoco_robot = Panda()
        # Creating a gripper
        self.gripper = gripper_factory('PandaGripper')
        self.mujoco_robot.add_gripper(self.gripper)
        # Setting robot position
        self.mujoco_robot.set_base_xpos([0, 0, 0])
        self.world.merge(self.mujoco_robot)

        # Setting table position
        self.mujoco_arena = TableArena()
        self.mujoco_arena.set_origin([0.8, 0, 0])
        self.world.merge(self.mujoco_arena)

        # Adding a ball
        self.sphere = BallObject(
            name="sphere",
            size=[0.04],
            rgba=[0, 0.5, 0.5, 1]).get_obj()
        self.sphere.set('pos', '1.0 0 1.0')
        self.world.worldbody.append(self.sphere)


        self.model = self.world.get_model(mode="mujoco")


if __name__ =='__main__':
    env = CustomExample()
    data = mujoco.MjData(env.model)     
    while data.time < 1:        
        mujoco.mj_step(env.model, data)     
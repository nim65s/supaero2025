import example_robot_data as robex
from supaero2024.meshcat_viewer_wrapper import MeshcatVisualizer

ROBOT_NAME = "solo12"
# ROBOT_NAME = 'ur5'

robot = robex.load(ROBOT_NAME)
viz = MeshcatVisualizer(robot, url="classical")
viz.display(robot.q0)

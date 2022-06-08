import numpy as np
import moveit_commander
from moveit_msgs.msg import DisplayTrajectory
import rospy
from std_msgs.msg import Header, Float64, Float64MultiArray, Int64


def dist_to_guess(p_base, guess):
    return np.sqrt((p_base.x - guess[0])**2 + (p_base.y - guess[1])**2 + (p_base.z - guess[2])**2)

def vector3ToNumpy(v):
    return np.array([v.x, v.y, v.z])

def move_ur5(move_group, robot, disp_traj_pub, input, plan=None, no_confirm=False):
    attempted = True

    if type(input) == list:
        move_group.set_joint_value_target(input)
    else:
        move_group.set_pose_target(input)

    if not plan:
        plan = move_group.plan()

    if no_confirm or check_valid_plan(disp_traj_pub, robot, plan):
        move_group.execute(plan, wait=True)
    else: 
        print("Plan is invalid!")
        attempted = False

    move_group.stop()
    move_group.clear_pose_targets()

    return attempted

def show_motion(disp_traj_pub, robot, plan):
    display_trajectory = DisplayTrajectory()
    display_trajectory.trajectory_start = robot.get_current_state()
    display_trajectory.trajectory.append(plan)
    disp_traj_pub.publish(display_trajectory)

def check_valid_plan(disp_traj_pub, robot, plan):
    run_flag = "d"

    while run_flag == "d":
        show_motion(disp_traj_pub, robot, plan)
        run_flag = raw_input("Valid Trajectory [y to run]? or display path again [d to display]:")

    return True if run_flag == "y" else False

def floatToMsg(data):
    force_msg = Float64()
    force_msg.data = data
    return force_msg

def intToMsg(data):
    force_msg = Int64()
    force_msg.data = data
    return force_msg

def floatArrayToMsg(data):
    array = Float64MultiArray()
    array.data = data
    return array
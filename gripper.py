from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output  as outputMsg

def open_gripper_msg():
    command = outputMsg.Robotiq2FGripper_robot_output()
    command.rPR = 0
    command.rACT = 1
    command.rGTO = 1
    command.rSP  = 255
    command.rFR  = 150

    return command

def close_gripper_msg():
    command = outputMsg.Robotiq2FGripper_robot_output()
    command.rPR = 255
    command.rACT = 1
    command.rGTO = 1
    command.rSP  = 255
    command.rFR  = 150

    return command

def activate_gripper_msg():
    command = outputMsg.Robotiq2FGripper_robot_output()
    command.rACT = 1
    command.rGTO = 1
    command.rSP  = 255
    command.rFR  = 150

    return command

def reset_gripper_msg():
    command = outputMsg.Robotiq2FGripper_robot_output()
    command.rACT = 0

    return command

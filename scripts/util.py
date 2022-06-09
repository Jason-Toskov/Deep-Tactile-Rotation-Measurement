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

def find_best_cart_grasp(self, data, choose_random=True):
        timeout_time = 15 # in seconds
        offset_dist = 0.1
        max_angle = 90
        final_grasp_pose = 0
        final_grasp_pose_offset = 0

        num_bad_angle = 0
        num_bad_plan = 0

        poses = []
        if choose_random:
            #Shuffle grasps
            data.grasps = filter(lambda x : x.score > self.threshold, data.grasps)
            random.shuffle(data.grasps)
            # random.shuffle(data.grasps)
            # rospy.loginfo('Grasps shuffled!')
        else:
            # Sort grasps by quality
            data.grasps.sort(key=lambda x : x.score, reverse=True)
            # rospy.loginfo("Grasps Sorted!")

        # Back up random list
        rand_grasps = copy.deepcopy(data.grasps)
        random.shuffle(rand_grasps)

        loop_start_time = rospy.Time.now()
        # loop through grasps from high to low quality
        for g_normal, g_rand in zip(data.grasps, rand_grasps):
            if rospy.is_shutdown():
                break

            # Take from whatever the option was (sorted/random) if not timed out
            time_taken = (rospy.Time.now() - loop_start_time).secs
            if time_taken < timeout_time:
                g = g_normal
                # rospy.loginfo("Using selected shuffle")
            elif time_taken < 3*timeout_time:
                g = g_rand
                # rospy.loginfo("Using random shuffle")
            else:
                rospy.loginfo("Max search time exceeded!")
                break
            
            # Get grasp pose from agile grasp outputs
            R = np.zeros((3,3))
            R[:, 0] = vector3ToNumpy(g.approach)
            R[:, 1] = vector3ToNumpy(g.axis)
            R[:, 2] = np.cross(vector3ToNumpy(g.approach), vector3ToNumpy(g.axis))

            q = Quaternion(matrix=R)
            position =  g.surface
            # rospy.loginfo("Grasp cam orientation found!")

            #Create poses for grasp and pulled back (offset) grasp
            p_base = PoseStamped()

            p_base.pose.position.x = position.x 
            p_base.pose.position.y = position.y 
            p_base.pose.position.z = position.z 

            p_base.pose.orientation.x = q[1]
            p_base.pose.orientation.y = q[2]
            p_base.pose.orientation.z = q[3]
            p_base.pose.orientation.w = q[0]

            p_base_offset = copy.deepcopy(p_base)
            p_base_offset.pose.position.x -= g.approach.x *offset_dist
            p_base_offset.pose.position.y -= g.approach.y *offset_dist
            p_base_offset.pose.position.z -= g.approach.z *offset_dist

            # Here we need to define the frame the pose is in for moveit
            p_base.header.frame_id = "base_link"
            p_base_offset.header.frame_id = "base_link"

            # Used for visualization
            poses.append(copy.deepcopy(p_base.pose))

            # Find angle between -z axis and approach
            approach_base = np.array([g.approach.x, g.approach.y, g.approach.z])
            approach_base = approach_base / np.linalg.norm(approach_base)
            theta_approach = np.arccos(np.dot(approach_base, np.array([0,0,-1])))*180/np.pi

            # rospy.loginfo("Grasp base orientation found")  

            # If approach points up, no good            
            if theta_approach < max_angle:
                (offset_plan, fraction) = self.move_group.compute_cartesian_path([p_base_offset.pose], 0.01, 0)
                if fraction != 1:
                    # Check final step
                    (final_plan, fraction) = self.move_group.compute_cartesian_path([p_base.pose], 0.01, 0)
                    if fraction != 1:

                        user_check = self.user_check_path(p_base_offset.pose, offset_plan)

                        if user_check:
                            # If so, we've found the grasp to use
                            final_grasp_pose = p_base
                            final_grasp_pose_offset = p_base_offset
                            rospy.loginfo("Final grasp found!")
                            # rospy.loginfo(" Angle: %.4f",  theta_approach)
                            # Only display the grasp being used
                            poses = [poses[-1]]
                            break
                        else:
                            rospy.loginfo("Invalid path")
                            num_bad_plan += 1
                    else:
                        rospy.loginfo("Invalid path")
                        num_bad_plan += 1
                else:
                    rospy.loginfo("Invalid path")
                    num_bad_plan += 1
            else:
                rospy.loginfo("Invalid angle of: " + str(theta_approach) + " deg")
                num_bad_angle += 1

        # Publish grasp pose arrows
        posearray = PoseArray()
        posearray.poses = poses
        posearray.header.frame_id = "base_link"
        self.pose_publisher.publish(posearray)

        #print("final_grasp_pose", final_grasp_pose)
        # rospy.loginfo("# bad angle: " + str(num_bad_angle))
        # rospy.loginfo("# bad plan: " + str(num_bad_plan))

        if not final_grasp_pose:
            offset_plan = 0
            rospy.loginfo("No valid grasp found!")

        return final_grasp_pose_offset, final_grasp_pose, offset_plan

def run_motion(self, state, final_grasp_pose_offset, plan_offset, final_grasp_pose):
    # Set based on state to either box
    # drop_joints = self.drop_joints_no_box[state]if self.no_boxes else self.drop_joints[state]
    drop_joints = [0.0, -1.6211016813861292, -1.9219277540790003, -1.166889492665426, 1.5740699768066406, -4.7985707418263246e-05]

    # Joint sequence
    joints_to_move_to = [self.move_home_joints, self.stable_test_joints, self.move_home_joints]

    # Move home
    self.move_group.set_start_state_to_current_state()
    self.move_to_joint_position(self.move_home_joints)
    rospy.sleep(0.2)

    # Grab object
    # successful_move = self.move_to_cartesian_position(final_grasp_pose_offset.pose, plan_offset) # Transition to cart
    offset_attempted = self.move_to_position(final_grasp_pose_offset, plan_offset)
    rospy.sleep(0.2)

    # self.move_to_cartesian_position(final_grasp_pose.pose)
    final_attempted = self.move_to_position(final_grasp_pose)
    rospy.sleep(0.2)
    self.command_gripper(close_gripper_msg())
    rospy.sleep(1)

    # Lift up
    self.move_to_position(self.lift_up_pose())
    rospy.sleep(0.2)

    attempted_grasp = True

    if offset_attempted and final_attempted:
        # Check grasp success
        # Success
        if self.check_grasp_success():
            rospy.loginfo("Robot has grasped the object!")
            success = 1
            # Stability check 
            rospy.loginfo("Checking stability...")
            for joints in joints_to_move_to:
                rospy.sleep(1)
                self.move_to_joint_position(joints)
            rospy.sleep(3)

            # Check if still in gripper (stable grasp)
            if self.check_grasp_success():
                stable_success = 1
            else:
                stable_success = 0

            # Drop object
            self.move_to_joint_position(drop_joints)
            self.command_gripper(open_gripper_msg())
            rospy.sleep(0.5)
        # Fail
        else:
            rospy.loginfo("Robot has missed/dropped object!")
            success = 0
            stable_success = 0
    else:
        attempted_grasp = False
        rospy.loginfo("Robot could not plan!")
        success = 0
        stable_success = 0

    # Move home    
    self.move_to_joint_position(self.move_home_joints)
    rospy.sleep(0.2)

    return success, stable_success, attempted_grasp
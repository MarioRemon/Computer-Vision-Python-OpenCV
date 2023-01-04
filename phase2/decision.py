import numpy as np
import time

# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!
    # Once we szare done we need to get back to the starting position
    if (Rover.samples_collected > 4):
        if(Rover.mapping > 95):
            print("GO TO START")
            dist_start = np.sqrt((Rover.pos[0] - Rover.start_pos[0]) ** 2 + (Rover.pos[1] - Rover.start_pos[1]) ** 2)
        # If we are in 10 meters steer to starting point

        # If we are in 3 meters just stop
            if dist_start < 10.0:
                print("10m From start")
                Rover.mode = 'stop'
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                return Rover
    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        # Check for Rover.mode status
        if Rover.mode == 'forward':
            # Check the extent of navigable terrain
            if len(Rover.nav_angles) >= Rover.stop_forward:
                # If mode is forward, navigable terrain looks good
                # and velocity is below max, then throttle
                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0
                Rover.brake = 0
                # Set steering to average angle clipped to the range +/- 15
                Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'
            # This will determine if the rover get stuck
            # Rover.stuck_time is a commulative frame counter that rover get stuck
            if (np.abs(Rover.vel) <= 0.05):
                Rover.stuck_time += 1
            # if it get can get out then fine, reset counter
            else:
                Rover.stuck_time = 0
            # if the rover get stuck, move backward in oppospite direction
            if Rover.stuck_time > 120:
                Rover.throttle = -Rover.throttle_set
                Rover.steer = -np.clip(np.mean(Rover.nav_angles * 180 / np.pi), -25, 25)
                time.sleep(0.5)

        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if len(Rover.nav_angles) < Rover.go_forward:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = -15 # Could be more clever here about which way to turn
                # If we're stopped but see sufficient navigable terrain in front then go!
                if len(Rover.nav_angles) >= Rover.go_forward:
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15,15 )
                    Rover.mode = 'forward'
                # If any sample detected, go to sample
        elif Rover.mode == 'goto_rock':
            # if the rover can pick up sample, stop and pick it up
            if Rover.near_sample:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
            # if cannot pick up, but the rover is getting close to the sample, start to brake
            elif Rover.vel > np.mean(Rover.nav_dists):
                Rover.throttle = 0
                Rover.brake = Rover.brake_set / 2
            # if the rover is not close to the sample yet, continue going with speed limit
            elif Rover.vel < Rover.max_vel / 2:
                Rover.throttle = Rover.throttle_set / 2
                Rover.brake = 0
            # if the rover is over the speed limit, brake slowly
            elif Rover.vel > Rover.max_vel / 2:
                Rover.throttle = 0
                Rover.brake = Rover.throttle_set / 3
            # --------------------------------------------------------------------------
            # This will determine if the rover get stuck
            # Rover.stuck_time is a commulative frame counter that rover get stuck
            if (np.abs(Rover.vel) <= 0.05):
                Rover.stuck_time += 1
            # if it get can get out then fine, reset counter
            else:
                Rover.stuck_time = 0
            # if the rover get stuck, move backward in oppospite direction
            if Rover.stuck_time > Rover.error_limit:
                Rover.throttle = -Rover.throttle_set
                Rover.steer = -np.clip(np.mean(Rover.nav_angles * 180 / np.pi), -25, 25)
                time.sleep(1)
            # --------------------------------------------------------------------------
            Rover.steer = np.clip(np.mean(Rover.nav_angles * 180 / np.pi), -25, 25)
            # if the sample is picked up, exit this mode
            if Rover.picking_up:
                Rover.mode = 'stop'

    # Just to make the rover do something
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0

    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
        # Enter mode 'stop' after picking up
        Rover.mode = 'stop'
    
    return Rover


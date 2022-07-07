rng(100,"twister")

mapData = load("uavMapCityBlock.mat");
omap = mapData.omap;

% Consider unknown spaces to be unoccupied
omap.FreeThreshold = omap.OccupiedThreshold;
show(omap)

startPose = [12 22 25 0.7 0.2 0 0.1];  % [x y z qw qx qy qz]
goalPose = [150 180 35 0.3 0 0.1 0.6];

% Plot the start and goal poses
hold on
scatter3(startPose(1),startPose(2),startPose(3),100,".r")
scatter3(goalPose(1),goalPose(2),goalPose(3),100,".g")
view([-31 63])
legend("","Start Position","Goal Position")
hold off

ss = stateSpaceSE3([-20 220;
                    -20 220;
                    -10 100;
                    inf inf;
                    inf inf;
                    inf inf;
                    inf inf]);
                
sv = validatorOccupancyMap3D(ss,Map=omap);
sv.ValidationDistance = 0.1;

planner = plannerRRTStar(ss,sv);
planner.MaxConnectionDistance = 50;
planner.GoalBias = 0.8;
planner.MaxIterations = 1000;
planner.ContinueAfterGoalReached = true;
planner.MaxNumTreeNodes = 10000;

[pthObj,solnInfo] = plan(planner,startPose,goalPose);

if (~solnInfo.IsPathFound)
    disp("No Path Found by the RRT, terminating example")
    return
end

% Plot map, start pose, and goal pose
show(omap)
hold on
scatter3(startPose(1),startPose(2),startPose(3),100,".r")
scatter3(goalPose(1),goalPose(2),goalPose(3),100,".g")

% Plot path computed by path planner
plot3(pthObj.States(:,1),pthObj.States(:,2),pthObj.States(:,3),"-g")
view([-31 63])
legend("","Start Position","Goal Postion","Planned Path")
hold off

waypoints = pthObj.States;
nWayPoints = pthObj.NumStates;

% Calculate the distance between waypoints
distance = zeros(1,nWayPoints);
for i = 2:nWayPoints
    distance(i) = norm(waypoints(i,1:3) - waypoints(i-1,1:3));
end

% Assume a UAV speed of 3 m/s and calculate time taken to reach each waypoint
UAVspeed = 3;
timepoints = cumsum(distance/UAVspeed);
nSamples = 100;

% Compute states along the trajectory
initialStates = minsnappolytraj(waypoints',timepoints,nSamples,MinSegmentTime=4,MaxSegmentTime=20,TimeAllocation=true,TimeWeight=100)';

% Plot map, start pose, and goal pose
show(omap)
hold on
scatter3(startPose(1),startPose(2),startPose(3),30,".r")
scatter3(goalPose(1),goalPose(2),goalPose(3),30,".g")

% Plot the waypoints
plot3(pthObj.States(:,1),pthObj.States(:,2),pthObj.States(:,3),"-g")

% Plot the minimum snap trajectory
plot3(initialStates(:,1),initialStates(:,2),initialStates(:,3),"-y")
view([-31 63])
legend("","Start Position","Goal Postion","Planned Path","Initial Trajectory")
hold off

% Check if the trajectory is valid
states = initialStates;
valid = all(isStateValid(sv,states));

while(~valid)
    % Check the validity of the states
    validity = isStateValid(sv,states);

    % Map the states to the corresponding waypoint segments
    segmentIndices = exampleHelperMapStatesToPathSegments(waypoints,states);

    % Get the segments for the invalid states
    % Use unique, because multiple states in the same segment might be invalid
    invalidSegments = unique(segmentIndices(~validity));

    % Add intermediate waypoints on the invalid segments
    for i = 1:size(invalidSegments)
        segment = invalidSegments(i);
        
        % Take the midpoint of the position to get the intermediate position
        midpoint(1:3) = (waypoints(segment,1:3) + waypoints(segment+1,1:3))/2;
        
        % Spherically interpolate the quaternions to get the intermediate quaternion
        midpoint(4:7) = slerp(quaternion(waypoints(segment,4:7)),quaternion(waypoints(segment+1,4:7)),.5).compact;
        waypoints = [waypoints(1:segment,:); midpoint; waypoints(segment+1:end,:)];

    end

    nWayPoints = size(waypoints,1);
    distance = zeros(1,nWayPoints);
    for i = 2:nWayPoints
        distance(i) = norm(waypoints(i,1:3) - waypoints(i-1,1:3));
    end
    
    % Calculate the time taken to reach each waypoint
    timepoints = cumsum(distance/UAVspeed);
    nSamples = 100;
    states = minsnappolytraj(waypoints',timepoints,nSamples,MinSegmentTime=2,MaxSegmentTime=20,TimeAllocation=true,TimeWeight=5000)';    
    
    % Check if the new trajectory is valid
    valid = all(isStateValid(sv,states));
   
end

% Plot map, start and goal pose
show(omap)
hold on
scatter3(startPose(1),startPose(2),startPose(3),30,".r")
scatter3(goalPose(1),goalPose(2),goalPose(3),30,".g")

% Plot the waypoints
plot3(pthObj.States(:,1),pthObj.States(:,2),pthObj.States(:,3),"-g")

% Plot the initial trajectory
plot3(initialStates(:,1),initialStates(:,2),initialStates(:,3),"-y")

% Plot the final valid trajectory
plot3(states(:,1),states(:,2),states(:,3),"-c")
view([-31 63])
legend("","Start Position","Goal Postion","Planned Path","Initial Trajectory","Valid Trajectory")
hold off
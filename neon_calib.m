function [positions_target, positions_eye] = calibrateEye(myscreen, task, recalibrateEyelink)
    % author : Josh Ryu (modified by HG)

    % screencolor = 0.5;
    screencolor = 0.0;
    % calibration parameters
    % r = [0.5, 3, 6, 9, 12];
    % a = 0:pi/6:2*pi;
    steady_time_thresh = 30; % in frames. have to fixate for this long.
    steady_deg_thresh = 0.5; % in degrees. threshold for eye being fixated

    % connect pupil labs
    pupil = true;
    
    if pupil
        global device
        ip = '10.35.125.54';
        device = pyrunfile("connect_pupillabs.py","device",ip=ip);
        if isempty(device) || isempty(device.phone_id)
            error("Pupil lab device is not connected.")
        end
    end
    

    % open screen
    myscreen.displayName = 'dell_wuTsai';
    myscreen.saveData = 1;
    myscreen = initScreen(myscreen);

    positions_target = [ 0,0; square_points(17.5); square_points(35) ];
    idx = randperm(size(positions_target,1));
    positions_target = positions_target(idx,:); % [n_position,2]
    positions_target = repmat(positions_target, 10,1);
    positions_eye = nan( [size(positions_target), steady_time_thresh+1] );

    n = 1; % stimulus number
    
    % [pos, postime] = mglEyelinkGetCurrentEyePos; % is this in image coordinates?
    %%%% instead of this, need to get data stream from pupil labs


    currpos = pos;
    buffpos = nan([2, steady_time_thresh+1]);
    steady = 0;
    toggle_skip = false;
    
    mglClearScreen(screencolor);
    while 1
        keycode = mglGetKeys;
        
        % Check if the 'g' key is pressed
        if keycode(6)
            if ~toggle_skip
                toggle_skip = true; % Key pressed, set toggle to true
                n = n + 1; % Skip to the next stimulus
                if n > size(positions_target, 1)
                    break % If this was the last stimulus, exit the loop
                end
            end
        else
            toggle_skip = false; % Key released, reset toggle
        end
        mglClearScreen(screencolor);

        
        if steady > steady_time_thresh
            % positions_eye(n,:) = pos; % record final position
            positions_eye(n,:,:) = buffpos;
            steady = 0;
            error = hypot(pos(1)-positions_target(n,1), pos(2)-positions_target(n,2));
            % sprintf('target position (%2.2f, %2.2f); eye position: (%2.2f, %2.2f); error: %2.2f',...
            %     positions_target(n,1), positions_target(n,2), pos(1), pos(2), error);
            disp(['Target position: (', num2str(positions_target(n, 1), '%.2f'), ', ', num2str(positions_target(n, 2), '%.2f'), ...
                  '); Eye position: (', num2str(pos(1), '%.2f'), ', ', num2str(pos(2), '%.2f'), ...
                  '); Error: ', num2str(error, '%.2f')]);
            n = n + 1; % go to next stimulus.
            if n > size(positions_target,1)
                break % done.
            end
        end
        
        [pos, postime] = mglEyelinkGetCurrentEyePos; % is this in image coordinates?

        % mglGluDisk(positions_target(n,1), positions_target(n,2), 0.2, [1 0 0, 1]);
        mglGluDisk(positions_target(n,1), positions_target(n,2), 0.2, [1 1 1, 1]);
        mglGluDisk(positions_target(n,1), positions_target(n,2), 0.1, [1 1 1, 1]);
        % disp(pos); disp(steady);
        
        if hypot(pos(1)-currpos(1),pos(2)-currpos(2)) < steady_deg_thresh
            steady = steady + 1;
            buffpos(:, steady) = pos;
        else
            % reset
            steady = 0;
            currpos = pos;
            buffpos = nan([2, steady_time_thresh+1]);
        end
        
        myscreen = tickScreen(myscreen,[]);     % flip screen
    end

end


function coords = square_points(len)

    half_len = len / 2;
    x = [-half_len, 0, half_len];
    y = [-half_len, 0, half_len];

    [X,Y] = meshgrid(x,y);
    coords = [X(:), Y(:)];
    coords(all(coords == 0, 2), :) = []; % remove origin

end
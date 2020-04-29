function computeBandPower(band, subs, condset, f_output)
% compute the specified band power by Hilbert transform

cd('c:\\topic_mind wandering\\3data')
load('pars.mat','times','srate')
load('pars_marker.mat', 'bands')

% set default
if nargin < 4 || isempty(f_output) 
    f_output = 'measure_matfile';
end

if nargin < 3 || isempty(condset)
    disp('Computing based on default conditions...')
    load('pars.mat', 'triggersets', 'tasks', 'states')
    defaultOn = 1;
    factors = {'state'};
    levels{1} = states;
else
     defaultOn = 0;
     triggersets = condset.triggersets;
     tasks       = condset.tasks;
     factors     = condset.factors;
     levels      = condset.levels; 
end

% get the band definition
id = find(strcmp({bands.name}, band));
measure = bands(id).name;
bandrange = bands(id).range;
chans = bands(id).chans;

baseline  = [-400 0];
stimOn    = [0 600];
baseIdx   = dsearchn(times', baseline');
stimOnIdx = dsearchn(times', stimOn');

% get level counts
nlevel = zeros(1, length(factors));  
for fi = 1:length(factors)
    nlevel(fi) = length(levels{fi});
end

% get level combos
if length(nlevel) > 1
    combos = zeros(prod(nlevel), length(nlevel));  % row = combo, column = factor, val = id in each factor
    i = length(nlevel);
    while i > 0
        if i == length(nlevel)
            combos(:,i) = repmat(1:nlevel(i), 1, size(combos,1)/nlevel(i));
        else
            temp = repmat(1:nlevel(i), prod(nlevel((i + 1):end)), 1);
            temp = temp(:)';
            combos(:,i) = repmat(temp, 1, size(combos,1)/length(temp));
        end
        i = i - 1;
    end 
else
    combos = [1:nlevel]';
end

n = length(tasks) * size(combos,1)* length(subs);
i = 0;
progressbar(['Computing power in ', band, ' band...'])

for taski = 1:length(tasks)
for comboi = 1:size(combos,1)
    
    task = tasks{taski};
    if ~defaultOn
        slicebase = ['{', num2str(taski), '}']; 
        triggerset = {};
        cond = '';
        for fi = 1:size(combos, 2)
            eval(['triggerset{fi} = triggersets', slicebase, '{', num2str(fi), '}{', num2str(combos(comboi,fi)), '};'])
            if fi ~= size(combos,2)
                cond = [cond, levels{fi}{combos(comboi, fi)}, '_'];
            else
                cond = [cond, levels{fi}{combos(comboi, fi)}];
            end
        end
    else
        triggers = triggersets{taski}{comboi};
        cond = states{comboi};
    end
    
    for sub = subs
        % get data
        if defaultOn
          [data, idx, rts] = select_trials(sub, triggers);
        else              
           args = {};
           for fi = 1:length(factors)
               args = [args, factors{fi}, triggerset{fi}];
           end
           [data, idx, rts] = select_trials2(sub, args);
        end
        
        if isempty(data)
            for chani = 1:length(chans)
                chan = chans(chani);
                subfolder = [measure, ' chan', num2str(chan)];
                varName = [measure,'_',task,'_',cond];
                file = [f_output, '\\', subfolder, '\\', num2str(sub), '.mat'];
                eval([varName,'= [];'])
                if exist(file,'file') == 2
                    save(file, varName, '-append')
                else
                    save(file, varName)
                end
            end  % chani
            i = i+1;
            progressbar(i/n)
            continue
        end
            
        if sub == subs(1)
            dataFilt = hilbertFilter(data(chans,:,:), srate, bandrange, 1);
        else 
            dataFilt = hilbertFilter(data(chans,:,:), srate, bandrange, 0);
        end

        powerAllChans = compute_power(dataFilt, baseIdx, 0);  % baseline correction: 'z', '%', 0 

        for chani = 1:length(chans)
            chan = chans(chani);
            power = powerAllChans(chani,:,:);
            mat = [idx, squeeze(mean(power(1,baseIdx(1):baseIdx(end),:),2)), squeeze(mean(power(1,stimOnIdx(1):stimOnIdx(end),:),2)), rts']; % colNames: idx, baseline, afterStim, rts 

            subfolder = [measure, ' chan', num2str(chan)];
            varName = [measure,'_',task,'_',cond];
            file = [f_output, '\\', subfolder, '\\', num2str(sub), '.mat'];
            eval([varName,'= mat;'])
            if exist(file,'file') == 2
                save(file, varName, '-append')
            else
                save(file, varName)
            end
        end  % chani
        i = i+1;
        progressbar(i/n)
    end  % sub
end  % comboi
end  % taski

end
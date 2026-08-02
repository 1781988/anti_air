function convert_matlab_tables(input_dir, output_dir)
% Convert MATLAB table-based radar MAT files to plain numeric MAT files.
% Usage: convert_matlab_tables('data/train', 'data/train_converted')

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
files = dir(fullfile(input_dir, '**', '*.mat'));
for i = 1:numel(files)
    source = fullfile(files(i).folder, files(i).name);
    data = load(source);
    names = fieldnames(data);
    converted = false;
    for j = 1:numel(names)
        value = data.(names{j});
        if istable(value) || istimetable(value)
            numericMask = varfun(@isnumeric, value, 'OutputFormat', 'uniform');
            if ~any(numericMask)
                warning('No numeric columns in %s', source);
                continue;
            end
            radar_data = table2array(value(:, numericMask)); %#ok<NASGU>
            variable_names = value.Properties.VariableNames(numericMask); %#ok<NASGU>
            destination = fullfile(output_dir, files(i).name);
            save(destination, 'radar_data', 'variable_names', '-v7');
            converted = true;
            fprintf('Converted %s -> %s\n', source, destination);
            break;
        end
    end
    if ~converted
        copyfile(source, fullfile(output_dir, files(i).name));
    end
end
end

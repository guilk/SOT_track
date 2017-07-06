function prepare_for_evaluation(src_root)

% src_root = '../../results/tracking_results';
dst_root = './results/results_TRE_CVPR13';
files = dir(src_root);
files = files(3:end);

for i = 1:length(files)
    results = cell(1,20);
    file_name = files(i).name;
    file_path = fullfile(src_root, file_name);
    splits = strsplit(file_name, '_');
    
    dst_path = fullfile(dst_root, strcat(lower(splits{1}),'_epoch_1_ignore.mat'));
    tracking_data = csvread(file_path);
    tracking_data(:,3) = tracking_data(:,3) - tracking_data(:,1);
    tracking_data(:,4) = tracking_data(:,4) - tracking_data(:,2);
    results{1}.res = tracking_data;
    results{1}.type = 'rect';
    results{1}.len = size(tracking_data,1);
    
    if strcmp(files(i).name, 'Tiger1_OTB_eval.txt')
        results{1}.startFrame = 1;
    elseif strcmp(files(i).name, 'David_OTB_eval.txt')
        results{1}.startFrame = 300;
    else
        results{1}.startFrame = 1;
    end
    disp(dst_path);
    
    save(dst_path, 'results');
end

end
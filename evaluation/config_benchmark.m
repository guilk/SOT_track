function seqs = config_benchmark(dataset)
seq_list_path = fullfile('./data_list',[dataset '_re.txt']);
seqs = {};

fid = fopen(seq_list_path);
tline = fgetl(fid);
index = 0;
while ischar(tline)
    splits = strsplit(tline,',');
    index = index + 1;
    seqs{index}.name = lower(splits{1});
    seqs{index}.path = splits{4};
    seqs{index}.startFrame = str2num(splits{2});
    seqs{index}.endFrame = str2num(splits{3});
    seqs{index}.nz = 4;
    seqs{index}.ext = 'jpg';
    seqs{index}.init_rect = [0 0 0 0];
    tline = fgetl(fid);
end
fclose(fid);
end
function filename = modify_prototxt(filename, num_blocks, noMeas)

fid = fopen(filename,'r');
i = 1;
tline = fgetl(fid);
A{i} = tline;
while ischar(tline)
    i = i+1;
    tline = fgetl(fid);
    A{i} = tline;
end
fclose(fid);
str1 = ['      dim:',' ',num2str(num_blocks)]; 
A{7} = sprintf('%s',str1);
str2 = ['      dim:',' ',num2str(noMeas)]; 
A{8} = sprintf('%s',str2);
% Write cell A into txt
fid = fopen([filename ], 'w');
for i = 1:numel(A)
    if A{i+1} == -1
        fprintf(fid,'%s', A{i});
        break
    else
        fprintf(fid,'%s\n', A{i});
    end
end
fclose(fid);
fig=figure(1);
clf;
for i = 1:num_control_units % (num_control_units+1):num_units
    days = 1:num_days;
    ys = control(i,:);
    hold on; plot(days, ys);
end
title("Control units");
filename = "data/synthetic/gpcontrol_" + SEED +".pdf";
set(fig, 'PaperPosition', [0 0 10 10]); %Position plot at left hand corner with width 5 and height 5.
set(fig, 'PaperSize', [10 10]); %Set the paper to have width 5 and height 5.
print(fig, filename, '-dpdf','-r300');

fig=figure(2);
clf;
for i = 1:num_treatment_units
    days = 1:num_days;
    ys = treat(i,:);
    hold on; plot(days, ys);
end
title("Treatment units");
filename = "data/synthetic/gptreat_" + int2str(SEED) +".pdf";
set(fig, 'PaperPosition', [0 0 10 10]); %Position plot at left hand corner with width 5 and height 5.
set(fig, 'PaperSize', [10 10]); %Set the paper to have width 5 and height 5.
print(fig, filename, '-dpdf','-r300');
load('data/control.csv');
num_control_units = size(control, 1);
num_days = size(control, 2);

ind = (~isnan(control));
[this_unit, this_time] = find(ind);
x = [this_time, ones(size(this_time)), this_unit];
y = control(ind);

load('data/treat.csv');
num_treatment_units = size(treat, 1);
num_units = num_control_units + num_treatment_units;

ind = (~isnan(treat));
[this_unit, this_time] = find(ind);
x = [x; [this_time, 2 * ones(size(this_time)), num_control_units + this_unit]];
y = [y; treat(ind)];

valid_days = unique(x(:, 1));

treatment_day = 89;

% for plotting
blues = [166, 206, 227; ...
          31, 120, 180] / 255;
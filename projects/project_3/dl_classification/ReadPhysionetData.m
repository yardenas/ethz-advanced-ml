    % This script parses the data from the PhysioNet 2017 Challenge and saves
% the data into PhysionetData.mat for quick and easy future use.

train_data = 'X_train.csv';
labels = 'y_train.csv';
test_data = 'X_test.csv';
Signals = readmatrix(train_data);
Signals = Signals(2:end, 2:end);
Labels = readmatrix(labels);
Labels = Labels(:, 2);
[SegmentedSignals, SegmentedLabels] = segmentSignals(Signals, Labels);
Labels = categorical(string(Labels));
% Save the variables to a MAT-file
save project_3_segmented_data.mat SegmentedSignals SegmentedLabels
save project_3_data.mat Signals Labels
data = csvread("ukf-nonlinear-out.csv"); 

calc_vel = [0; diff(data(:,2))]; 
plot(
	data(:,1), data(:,1), 
	data(:,1), data(:,2), 
	data(:,1), data(:,3), 
	data(:,1), data(:,4),
	data(:,1), data(:,5),
	data(:,1), calc_vel
); 

input("Press any key"); 

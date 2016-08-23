data = csvread("ukf-nonlinear-out.csv"); 

plot(
	data(:,1), data(:,2), 
	data(:,1), data(:,3), 
	data(:,1), data(:,4),
	data(:,1), data(:,5)
); 

input("Press any key"); 

%% Task 0: Warming-up
clear ; close all; clc

%% 1. Create a scalar (matrix, structure)
 %Create a scalar
a=5;
 %Create 2 matrices
A=[1 2 3;4 5 6; 7 8 9]
B=zeros(3);
for i=1:3
    for j=1:3
        B(i,j)=i+j;
    end
end
  %return an element
A(1,2)
  %return a column
A(:,2)
  %return a raw
A(1,:)
  %transpose a matrix
AT=A'
  %product 2 matrices
C=A*B
  %element-wise product of 2 matrices
D=A.*B
  %merge 2 matrices
E=[A B]
 %Create a structure
F=struct('name','Alexey','age',39,'height',182,'weight',83)
  %return an element
  G=F.age
  
%% 2. Create a function
[sum,dif] = myFunc(A,B)

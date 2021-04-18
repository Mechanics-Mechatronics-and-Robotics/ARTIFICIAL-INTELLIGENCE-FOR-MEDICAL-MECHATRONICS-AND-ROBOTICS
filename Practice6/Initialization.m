clear
clc
close all

%Whant to know more?
%Check the following links:
%https://www.mathworks.com/help/reinforcement-learning/ug/train-dqn-agent-to-balance-cart-pole-system.html
%https://www.mathworks.com/help/reinforcement-learning/ug/train-dqn-agent-to-swing-up-and-balance-pendulum.html

%The DQN agent approximates the long-term reward, 
%given observations and actions, using a value-function critic.

%For this environment:
%The upward balanced pendulum position is 0 radians, 
% and the downward hanging position is pi radians
%The pendulum starts upright with an initial angle of +/- 0.05 radians
%The force action signal from the agent to the environment
%is from -10 to 10 N
%The observations from the environment are the position and velocity 
%of the cart, the pendulum angle, and its derivative
%The episode terminates if the pole is more than "thetaThreshold" radians
%from vertical, or the cart moves more than "dispThreshold" from 
%the original position
%A reward of +1 is provided for every time step that the pole remains 
%upright. A penalty of -5 is applied when the pendulum falls.

%You will need to finish the CartPole.slx file
%% 0. Settings
curDir = pwd;
saveDir = 'savedAgents';
cd(saveDir)
savePath=cd;
cd(curDir)

%Mechanics
cartSize=[0.02 0.01 0.01];% cart size,m 
poleSize=[0.01 0.01 0.5];% pole size,m
cartMass=1;%cart mass, kg
poleMass=0.1;%pole mass,kg
rhoCart=cartMass/(cartSize(1)*cartSize(2)*cartSize(3));%density, kg/m^3
rhoPole=poleMass/(poleSize(1)*poleSize(2)*poleSize(3));%density, kg/m^3

%Initializations and Limits
initangle=0.1*(rand()-0.5)%rand value in the  interval of -0.05:0.05, rad
thetaThreshold=0.2;%+- max. angle of the pole rotation, rad
dispThreshold=5*cartSize(1);%+- max dispacement of the cart, m
totalTime=5;
totalTimeVis=15;%visualization time when check the trained agent, s
sampleTime=1e-2;
%Ts=5e-3;%control sample time

%The RL settings
nObs=[4 1];%number of observations
nAct=1;%number of actions
conSig=[-10 0 10];%control signals
r=1;%reward
penalty=-5;%penalty
criticOpts = rlRepresentationOptions('LearnRate',0.001,...
    'GradientThreshold',1);
agentOpts = rlDQNAgentOptions(...
    'SampleTime',sampleTime,...
    'UseDoubleDQN',false, ...    
    'TargetSmoothFactor',1, ...
    'TargetUpdateFrequency',10, ...   
    'ExperienceBufferLength',100000, ...
    'DiscountFactor',0.99, ...
    'MiniBatchSize',256);
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',1000, ...
    'MaxStepsPerEpisode',ceil(totalTime/sampleTime), ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',0.95*totalTime/sampleTime); 

numFCL=24;%number of neurons in the FC-layers

doTraining = true;% true,false

%% Create the DQN agent

mdl = 'CartPole';
%Assign the agent block path information, and create rlNumericSpec and rlFiniteSetSpec objects for the observation and action information.
agentBlk = [mdl '/RL Agent'];
obsInfo = rlNumericSpec(nObs);%([nObs 1])
actInfo = rlFiniteSetSpec(conSig);

obsInfo.Name = 'Observations';
actInfo.Name = 'Force';

%Create the reinforcement learning environment for the Simulink model using information extracted in the previous steps.
env = rlSimulinkEnv(mdl,agentBlk,obsInfo,actInfo)

%Fix the random generator seed for reproducibility.
rng(0)

%Create a deep neural network with one input (the 4-dimensional observed state) and one output vector with two elements (one for the 10 N action, another for the –10 N action). For more information on creating value-function representations based on a neural network
dnn = [
    featureInputLayer(obsInfo.Dimension(1),'Normalization','none','Name','state')
    fullyConnectedLayer(numFCL,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(numFCL, 'Name','CriticStateFC2')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(length(actInfo.Elements),'Name','output')];

% figure
% plot(layerGraph(dnn))

%Create the ctitic representation using the specified neural network and options. 
critic = rlQValueRepresentation(dnn,obsInfo,actInfo,...
    'Observation',{'state'},criticOpts);

%Create the DQN agent using the specified critic representation and agent options. For more information, see rlDQNAgent.
agent = rlDQNAgent(critic,agentOpts);

%% Download the Simulink model
open_system(mdl)

%%
if doTraining    
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
    cd(saveDir)
    save(['trainedAgent_2D_' datestr(now,'mm_DD_YYYY_HHMM')],'agent','trainingStats');
    cd(curDir)
else
    % Load the pretrained agent for the example.
    cd(saveDir)
    load('trainedAgent_2D_04_18_2021_1455.mat','agent')
    cd(curDir)
end

%% Visualization
simOptions = rlSimulationOptions('MaxSteps',ceil(50/sampleTime));
experience = sim(env,agent,simOptions);

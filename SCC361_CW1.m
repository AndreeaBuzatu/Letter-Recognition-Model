load("emnist-letters.mat")

%split the data into train and test
train_features = double(dataset.train.images());
train_labels = dataset.train.labels();

test_features = double(dataset.test.images());
test_labels = dataset.test.labels();

%K-Nearest Neigbour Classification
%train the KNN model using the command
knn_model = fitcknn(train_features,train_labels);

%predict the label of the new set of observations
knn_predictions = predict(knn_model,test_features);

%accuracy of the model
accuracy_knn = sum(knn_predictions==test_labels)/length(test_labels);
%0.8582

%performance loss
knn_loss = resubLoss(knn_model);
%0

%confusion matrix
figure
knn_CM = confusionchart(test_labels,knn_predictions);


%----------------------------------------------


%train the Decision Tree model using the command
dtree_model = fitctree(train_features,train_labels);

%predict the label of the new set of observations
dtree_predictions = predict(dtree_model,test_features);

%accuracy of the model
accuracy_dtree = sum(dtree_predictions==test_labels)/length(test_labels);
%0.7039

%performance loss
dtree_loss = resubLoss(dtree_model);
%0.0859

%confusion matrix
figure
dtree_CM = confusionchart(test_labels,dtree_predictions);


%----------------------------------------------


%train the Naive Bayes model using the command
nb_model = fitcnb(train_features,train_labels,"DistributionNames","mn");

%predict the label of the new set of observations
nb_predictions = predict(nb_model,test_features);

%accuracy of the model
accuracy_nb = sum(nb_predictions==test_labels)/length(test_labels);
%0.5783

%performance loss
nb_loss = resubLoss(nb_model);
%0.4214

%confusion matrix
figure
nb_CM = confusionchart(test_labels,nb_predictions);


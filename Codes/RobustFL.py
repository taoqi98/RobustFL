import numpy as np

def detection_accuracy(detected_clients,malicious_clients,user_indexs):
    detected_clients = set(detected_clients)
    user_indexs = set(user_indexs.tolist())
    P = len(detected_clients&malicious_clients)/(len(detected_clients)+10**(-6))
    if len(detected_clients) == 0 and len(user_indexs&malicious_clients) ==0:
        P = 1
    R = len(detected_clients&malicious_clients)/(len(user_indexs&malicious_clients)+10**(-6))
    if len(detected_clients) == 0 and len(user_indexs&malicious_clients) ==0:
        R = 1
    return P,R


def RobustFL(i,user_num,model,unlabeled_images,train_images,train_labels,train_users,malicious_clients,intre_epoch=3):
    user_indexs = np.random.randint(len(train_users),size=(user_num,))

    old_weights = model.get_weights()
    
    all_predictions = []
    all_updates = []
    
    for ui in range(len(user_indexs)):
        sample_indexs = train_users[user_indexs[ui]]
        x = train_images[sample_indexs]
        y = train_labels[sample_indexs]
        for i in range(intre_epoch):
            model.fit(x,y,verbose=0)
        weights = model.get_weights()
        
        local_predictions = model.predict(unlabeled_images)
        all_predictions.append(local_predictions)
        model.set_weights(old_weights)
        
    all_predictions = np.array(all_predictions)
    avg_predictions = all_predictions.mean(axis=0)
    
    prediction_bias = np.abs(all_predictions-avg_predictions.reshape((1,all_predictions.shape[1],all_predictions.shape[2]))).sum(axis=-1).mean(axis=-1)
    prediction_detected_client = set(user_indexs[np.where(prediction_bias>prediction_bias.mean())].tolist())

    if i>40:
        availabel_clients = []
        for j in range(len(user_indexs)):
            uix = user_indexs[j]
            if not uix in prediction_detected_client:
                availabel_clients.append(j)
        if len(availabel_clients)>len(user_indexs)//2:
            availabel_clients = np.array(availabel_clients)
            avg_predictions = all_predictions[availabel_clients].mean(axis=0)
            
    entropy = (-avg_predictions*np.log(avg_predictions+10**(-6))).sum(axis=-1)

    
    if i >5:
        index = np.where(entropy<0.3)[0]
        model.fit(unlabeled_images[index],avg_predictions[index])
    else:
        model.fit(unlabeled_images,avg_predictions)
        
    return detection_accuracy(prediction_detected_client,malicious_clients,user_indexs)
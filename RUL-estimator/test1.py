

>> model = Sequential()
model.add(Embedding(input_dim = num_words, output_dim =EMBEDDING_DIM , input_length=MAX_SEQUENCE_LENGTH,
             weights=[embedding_matrix],trainable=False))
# The model will take as input an integer matrix of size (batch,input_length), and the largest integer (i.e. word index) in the input  
# should be no larger than 999 (vocabulary size).  
# Now model.output_shape is (None, 10, 64), where `None` is the batch dimension.  

print("Building the model")
model.add(Conv1D(no_of_filter=128,filter_size=3,activation='relu'))	# then model.add(BatchNormalization())
model.add(MaxPooling1D(3) )
model.add(Conv1D(128,3,activation='relu'))
model.add(MaxPooling1D(3) )
model.add(Conv1D(128,3,activation='relu'))
model.add( GlobalMaxPooling1D() )

Why globalmaxpooling, we have a time series and we do not care how long the time series, we just take the maximum of the series in each dimension 
(T is the sequence length and M is the number of features, i.e., scanning i/p in time and know when the most significant feature occurred)

model.add( Dense(len(possible_labels),activation='sigmoid') )

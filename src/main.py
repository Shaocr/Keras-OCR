from model.CRNN import CRNN

C = CRNN()
Trainmodels,PredictModel = C.build((None,32,1))
adadelta = Adadelta(lr=0.05)
Trainmodels.compile(loss=lambda y_true, y_pred: y_pred , optimizer=adadelta)
Trainmodels.summary()
Trainmodels.fit_generator(
    C.generateBacthData(32),
    epochs=3,
    steps_per_epoch=100,
    verbose=1)

labels_pred = PredictModel.predict_on_batch(next(C.generate_test_data(32)))

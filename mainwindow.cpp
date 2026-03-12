#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <opencv2/dnn.hpp>

#include <QFileDialog>
#include <QCoreApplication>
#include <QRandomGenerator>
#include <QMessageBox>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    ui->treeView->setRootIsDecorated(false);
    ui->treeView->setIndentation(0);
    ui->treeView->setHeaderHidden(true);

    PyProcess = new QProcess(this);
    PyProcess->setProcessChannelMode(QProcess::MergedChannels);

    connect(PyProcess, &QProcess::readyReadStandardOutput, this, &MainWindow::readPythonOutput);
    connect(PyProcess, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &MainWindow::onProcessFinished);

    ui->progressBar->setVisible(false);

    fileModel = new QFileSystemModel(this);
    fileModel->setFilter(QDir::NoDotAndDotDot | QDir::Files);

    QStringList filtres;
    filtres << "*.png" << "*.jpg" << "*.jpeg";
    fileModel->setNameFilters(filtres);
    fileModel->setNameFilterDisables(false);

    QString appDir = QCoreApplication::applicationDirPath();
    QString cascadePath = appDir + "/haarcascade_frontalface_default.xml";
    if (!faceCascade.load(cascadePath.toStdString())) {
        qDebug() << "ERROR: Can't load Haar Cascade from:" << cascadePath;
        ui->btnSelectImage->setEnabled(false);
    } else {
        qDebug() << "Face Detector loaded successfully!";
    }

    QDir dir(appDir);
    QStringList modelFilters;
    modelFilters << "model_keypoints*.onnx";
    QStringList files = dir.entryList(modelFilters, QDir::Files);

    if (!files.isEmpty())
    {
        currentModelPath = appDir + "/" + files.first();
        QFileInfo fileInfo(currentModelPath);
        ui->modelNameLabel->setText(fileInfo.fileName());
        ui->modelNameLabel->setStyleSheet("color: #228B22; font-weight: bold; font-size: 14px;");
        qDebug() << "Auto-loaded keypoints model:" << currentModelPath;
    }
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::setModelAsActive(const QString &path)
{
    if(path.isEmpty()) return;

    currentModelPath = path;

    QFileInfo fileInfo(path);
    ui->modelNameLabel->setText(fileInfo.fileName());
    ui->modelNameLabel->setStyleSheet("color: #228B22; font-weight: bold; font-size: 14px;");
}

void MainWindow::setModelZNM1NameLabel(const QString &path)
{
    if(path.isEmpty()) return;

    currentModelZNM1Path = path;

    QFileInfo fileInfo(path);
    ui->modelZNM1NameLabel->setText(fileInfo.fileName());
    ui->modelZNM1NameLabel->setStyleSheet("color: #228B22; font-weight: bold; font-size: 14px;");
}

void MainWindow::setModelZNM2NameLabel(const QString &path)
{
    if(path.isEmpty()) return;

    currentModelZNM2Path = path;

    QFileInfo fileInfo(path);
    ui->modelZNM2NameLabel->setText(fileInfo.fileName());
    ui->modelZNM2NameLabel->setStyleSheet("color: #228B22; font-weight: bold; font-size: 14px;");
}

void MainWindow::on_btnSelectModel_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Select ONNX Model", "", "ONNX Models (*onnx)");
    if(fileName.isEmpty()) return;

    if (!fileName.contains("model_keypoints"))
    {
        QMessageBox::warning(this, "Помилка", "Будь ласка, оберіть правильну модель! Назва файлу має бути наприклад такою - 'model_keypoints_Accuracy_080.onnx'.");
        return;
    }

    setModelAsActive(fileName);

    if (!currentImagePath.isEmpty())
        processAndDisplayImage(currentImagePath);
}

void MainWindow::on_btnSelectZNM1Model_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Select ZNM1 Model", "", "ONNX Models (*onnx)");
    if (fileName.isEmpty()) return;

    if (!fileName.contains("ZNM1Keypoints"))
    {
        QMessageBox::warning(this, "Помилка", "Будь ласка, оберіть правильну модель! Назва має містити 'ZNM1Keypoints'.");
        return;
    }

    setModelZNM1NameLabel(fileName);
    predictEmotion();
}

void MainWindow::on_btnSelectZNM2Model_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Select ZNM2 Model", "", "ONNX Models (*onnx)");
    if (fileName.isEmpty()) return;

    if (!fileName.contains("ZNM2Photo"))
    {
        QMessageBox::warning(this, "Помилка", "Будь ласка, оберіть правильну модель! Назва має містити 'ZNM2Photo'.");
        return;
    }

    setModelZNM2NameLabel(fileName);

    if (!currentImagePath.isEmpty()) processAndDisplayImage(currentImagePath);
}

void MainWindow::onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    if(exitCode == 0 && exitStatus == QProcess::NormalExit)
    {
        if (currentProcessType == ProcessKeypoints)
            setModelAsActive(currentModelPath);
        else if (currentProcessType == ProcessZNM1)
        {
            setModelZNM1NameLabel(currentModelZNM1Path);
            predictEmotion();
        }
        else if (currentProcessType == ProcessZNM2)
        {
            setModelZNM2NameLabel(currentModelZNM2Path);
            predictEmotionZNM2();
        }
    }
    else
    {
        ui->progressBar->setFormat("Error / Stopped");
    }

    currentProcessType = ProcessNone;

    QTimer::singleShot(2000, this, [this]()
                       {
                           ui->progressBar->setVisible(false);
                           ui->progressBar->setValue(0);
                           ui->progressBar->setFormat("%p%");
                       });
}

void MainWindow::readPythonOutput()
{
    QByteArray InData = PyProcess->readAllStandardOutput();
    QString text = QString::fromLocal8Bit(InData);
    QStringList lines = text.split("\n");

    for(const QString line : lines)
    {
        QString cleanLine = line.trimmed();
        if(cleanLine.isEmpty())
        {
            continue;
        }

        if(cleanLine.startsWith("CURRENT_LEARN_PROGRESS:"))
        {
            QStringList elems = cleanLine.split(" ", Qt::SkipEmptyParts);

            if(elems.size() >= 4)
            {
                double targetAcc = ui->targetAccSpinBox->value();
                double currentAcc = elems[1].toDouble();
                int epoch = elems[3].toInt();
                double startAcc = 0.3;

                double progress = 0.0;
                if(targetAcc > startAcc)
                {
                    progress = (currentAcc - startAcc) / (targetAcc - startAcc);
                }
                int progressValue = static_cast<int>(progress * 100);
                if (progressValue < 0) progressValue = 0;
                if (progressValue > 100) progressValue = 100;

                ui->progressBar->setValue(progressValue);

                QString progressText = QString("Epochs passed: %1 / Current accuracy: %2")
                                           .arg(epoch).arg(currentAcc, 0, 'f', 4);

                ui->progressBar->setFormat(progressText);
                ui->progressBar->setAlignment(Qt::AlignCenter);
            }
        }
        else
        {
            qDebug() << "[Python]:" << cleanLine;
        }
    }
}

void MainWindow::on_btnTrainModel_clicked()
{
    if (PyProcess->state() == QProcess::Running) {
        return;
    }

    QString csvPath = QFileDialog::getOpenFileName(this, "Select Dataset", "", "CSV Files (*.csv)");
    if(csvPath.isEmpty()) return;

    ui->progressBar->setVisible(true);
    ui->progressBar->setValue(0);

    QString targetAcc = QString::number(ui->targetAccSpinBox->value());

    QString targetAccFName = targetAcc;
    targetAccFName.replace(",", "");
    targetAccFName.replace(".", "");

    currentModelPath = "model_keypoints_Accuracy_" + targetAccFName + ".onnx";

    currentProcessType = ProcessKeypoints;

    QStringList args;
    args << "train_keypoints_model.py" << csvPath << targetAcc << currentModelPath;

    PyProcess->start("python", args);
}

void MainWindow::on_btnTrainZNM1Model_clicked()
{
    if (PyProcess->state() == QProcess::Running) {
        return;
    }

    QString csvPath = QFileDialog::getOpenFileName(this, "Select Dataset", "", "CSV Files (*.csv)");
    if(csvPath.isEmpty()) return;

    ui->progressBar->setVisible(true);
    ui->progressBar->setValue(0);

    QString targetAcc = QString::number(ui->targetAccZnm1SpinBox->value());

    QString targetAccFName = targetAcc;
    targetAccFName.replace(",", "");
    targetAccFName.replace(".", "");

    currentModelPath = "model_ZNM1Keypoints_Accuracy_" + targetAccFName + ".onnx";

    currentProcessType = ProcessZNM1;

    QStringList args;
    args << "train_znm1.py" << csvPath << targetAcc << currentModelPath;

    PyProcess->start("python", args);
}

void MainWindow::on_btnTrainZNM2Model_clicked()
{
    if (PyProcess->state() == QProcess::Running) return;

    QString csvPath = QFileDialog::getOpenFileName(this, "Select Dataset", "", "CSV Files (*.csv)");
    if(csvPath.isEmpty()) return;

    ui->progressBar->setVisible(true);
    ui->progressBar->setValue(0);

    QString targetAcc = QString::number(ui->targetAccZnm2SpinBox->value());

    QString targetAccFName = targetAcc;
    targetAccFName.replace(",", "");
    targetAccFName.replace(".", "");

    currentModelZNM2Path = "model_ZNM2Photo_Accuracy_" + targetAccFName + ".onnx";

    currentProcessType = ProcessZNM2;

    QStringList args;
    args << "train_znm2.py" << csvPath << targetAcc << currentModelZNM2Path;

    PyProcess->start("python", args);
}

void MainWindow::processAndDisplayImage(const QString &imagePath, bool applyJitter)
{
    if (PyProcess->state() == QProcess::Running) return;
    if (imagePath.isEmpty()) return;

    currentImagePath = imagePath;

    ui->predictedEmotionLabel->setText("Emotion: -");
    ui->predictedEmotionZNM2Label->setText("Emotion: -");
    currentKeypoints.clear();
    currentFaceCrop.release();

    try
    {
        cv::Mat originalImage = cv::imread(imagePath.toStdString());
        if (originalImage.empty()) return;

        cv::Mat displayImage = originalImage.clone();
        cv::cvtColor(displayImage, displayImage, cv::COLOR_BGR2RGB);

        cv::Mat grayImage;
        cv::cvtColor(originalImage, grayImage, cv::COLOR_BGR2GRAY);

        std::vector<cv::Rect> faces;
        faceCascade.detectMultiScale(grayImage, faces, 1.1, 4, 0, cv::Size(20, 20));

        cv::Rect faceRect;

        if (!faces.empty())
        {
            faceRect = faces[0];
        }
        else
        {
            faceRect = cv::Rect(0, 0, originalImage.cols, originalImage.rows);
        }

        if (applyJitter)
        {
            int dx = QRandomGenerator::global()->bounded(-5, 6) * faceRect.width / 100;
            int dy = QRandomGenerator::global()->bounded(-5, 6) * faceRect.height / 100;
            int dw = QRandomGenerator::global()->bounded(-5, 6) * faceRect.width / 100;
            int dh = QRandomGenerator::global()->bounded(-5, 6) * faceRect.height / 100;

            faceRect.x += dx;
            faceRect.y += dy;
            faceRect.width += dw;
            faceRect.height += dh;

            if (faceRect.x < 0) faceRect.x = 0;
            if (faceRect.y < 0) faceRect.y = 0;
            if (faceRect.x + faceRect.width > originalImage.cols) faceRect.width = originalImage.cols - faceRect.x;
            if (faceRect.y + faceRect.height > originalImage.rows) faceRect.height = originalImage.rows - faceRect.y;
        }

        cv::Mat faceCrop = grayImage(faceRect);

        cv::resize(faceCrop, currentFaceCrop, cv::Size(48, 48));

        cv::Mat blob;
        cv::dnn::blobFromImage(faceCrop, blob, 1.0/255.0, cv::Size(96, 96), cv::Scalar(), false, false);

        if (!currentModelPath.isEmpty())
        {
            cv::dnn::Net net = cv::dnn::readNetFromONNX(currentModelPath.toStdString());
            net.setInput(blob);
            cv::Mat output = net.forward();

            for (int i = 0; i < output.cols; i++)
                currentKeypoints.push_back(output.at<float>(0, i));

            if (ui->tabWidget->currentIndex() == 0)
            {
                cv::rectangle(displayImage, faceRect, cv::Scalar(0, 255, 0), 2);

                for (size_t i = 0; i < currentKeypoints.size(); i += 2)
                {
                    float x_crop = currentKeypoints[i] * faceRect.width;
                    float y_crop = currentKeypoints[i+1] * faceRect.height;

                    float x_original = x_crop + faceRect.x;
                    float y_original = y_crop + faceRect.y;

                    cv::circle(displayImage, cv::Point(x_original, y_original), 3, cv::Scalar(0, 255, 0), -1);
                }
            }
        }

        if (!currentKeypoints.empty() && !currentModelZNM1Path.isEmpty())
            predictEmotion();

        if (!currentFaceCrop.empty() && !currentModelZNM2Path.isEmpty())
            predictEmotionZNM2();

        QImage qImg(displayImage.data, displayImage.cols, displayImage.rows, displayImage.step, QImage::Format_RGB888);
        ui->imageLabel->setPixmap(QPixmap::fromImage(qImg).scaled(ui->imageLabel->size(), Qt::KeepAspectRatio));

    }
    catch (const cv::Exception& e)
    {
        qDebug() << "OPENCV ERROR:" << e.what();
    }
}

void MainWindow::on_btnSelectImage_clicked()
{
    QString SelectedImagePath = QFileDialog::getOpenFileName(this, "Select Image", "", "Images (*.png *.jpg *.jpeg)");
    if(SelectedImagePath.isEmpty())
        return;

    currentImagePath = SelectedImagePath;

    QFileInfo fileInfo(SelectedImagePath);
    QString folderPath = fileInfo.absolutePath();

    QModelIndex rootIndex = fileModel->setRootPath(folderPath);
    if(!ui->treeView->model())
    {
        ui->treeView->setModel(fileModel);
        for(int i = 1; i < fileModel->columnCount(); ++i)
            ui->treeView->hideColumn(i);
    }
    ui->treeView->setRootIndex(rootIndex);

    processAndDisplayImage(currentImagePath);
}

void MainWindow::on_btnRedefinePoints_clicked()
{
    if (!currentImagePath.isEmpty())
        processAndDisplayImage(currentImagePath, true);
}

void MainWindow::on_treeView_doubleClicked(const QModelIndex &index)
{
    QString path = fileModel->filePath(index);
    currentImagePath = path;
    processAndDisplayImage(currentImagePath);
}

void MainWindow::predictEmotion()
{
    if (currentModelZNM1Path.isEmpty()) return;
    if (currentKeypoints.size() != 30) return;

    try
    {
        cv::dnn::Net net = cv::dnn::readNetFromONNX(currentModelZNM1Path.toStdString());

        cv::Mat inputBlob(1, 30, CV_32F, currentKeypoints.data());

        net.setInput(inputBlob);
        cv::Mat output = net.forward();

        cv::Point classIdPoint;
        double confidence;
        cv::minMaxLoc(output, nullptr, &confidence, nullptr, &classIdPoint);

        int emotionId = classIdPoint.x;

        QStringList emotions = {"Anger", "Contempt", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"};

        if (emotionId >= 0 && emotionId < emotions.size())
            ui->predictedEmotionLabel->setText("Emotion: " + emotions[emotionId]);

    }
    catch (const cv::Exception& e)
    {
        qDebug() << "ONNX ZNM1 ERROR:" << e.what();
    }
}

void MainWindow::predictEmotionZNM2()
{
    if (currentModelZNM2Path.isEmpty()) return;
    if (currentFaceCrop.empty()) return;

    try
    {
        cv::dnn::Net net = cv::dnn::readNetFromONNX(currentModelZNM2Path.toStdString());

        cv::Mat blob = cv::dnn::blobFromImage(currentFaceCrop, 1.0/255.0, cv::Size(48, 48), cv::Scalar(), false, false);

        net.setInput(blob);
        cv::Mat output = net.forward();

        cv::Point classIdPoint;
        double confidence;
        cv::minMaxLoc(output, nullptr, &confidence, nullptr, &classIdPoint);

        int emotionId = classIdPoint.x;

        QStringList emotions = {"Anger", "Contempt", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"};

        if (emotionId >= 0 && emotionId < emotions.size())
            ui->predictedEmotionZNM2Label->setText("Emotion: " + emotions[emotionId]);
    }
    catch (const cv::Exception& e)
    {
        qDebug() << "ONNX ZNM2 ERROR:" << e.what();
    }
}

void MainWindow::on_tabWidget_currentChanged(int index)
{
    if (!currentImagePath.isEmpty())
        processAndDisplayImage(currentImagePath);
}


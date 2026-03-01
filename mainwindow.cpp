#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <opencv2/dnn.hpp>

#include <QFileDialog>
#include <QCoreApplication>
#include <QRandomGenerator>

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
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::setModelAsActive(const QString &path)
{
    if(path.isEmpty())
        return;

    currentModelPath = path;

    QFileInfo fileInfo(path);
    ui->modelNameLabel->setText(fileInfo.fileName());
    //ui->modelNameLabel->setStyleSheet("color: green; font-weight: bold;");
    ui->modelNameLabel->setStyleSheet("color: #228B22; font-weight: bold; font-size: 14px;");
}

void MainWindow::on_btnSelectModel_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Select ONNX Model", "", "ONNX Models (*onnx)");
    if(fileName.isEmpty())
    {
        return;
    }
    setModelAsActive(fileName);
}

void MainWindow::onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    if(exitCode == 0 && exitStatus == QProcess::NormalExit)
    {
        ui->progressBar->setValue(100);
        setModelAsActive(currentModelPath);
    }
    else
    {
        ui->progressBar->setFormat("Error / Stopped");
    }

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

    currentModelPath = "model_Accuracy_" + targetAccFName + ".onnx";

    QStringList args;
    args << "train_model.py" << csvPath << targetAcc << currentModelPath;

    PyProcess->start("python", args);
}


void MainWindow::processAndDisplayImage(const QString &imagePath, bool applyJitter)
{
    if (PyProcess->state() == QProcess::Running) return;
    if (currentModelPath.isEmpty() || imagePath.isEmpty()) return;

    currentImagePath = imagePath;

    try
    {
        cv::dnn::Net net = cv::dnn::readNetFromONNX(currentModelPath.toStdString());

        cv::Mat originalImage = cv::imread(imagePath.toStdString());
        if (originalImage.empty()) return;

        cv::Mat grayImage;
        cv::cvtColor(originalImage, grayImage, cv::COLOR_BGR2GRAY);
        cv::cvtColor(originalImage, originalImage, cv::COLOR_BGR2RGB);

        std::vector<cv::Rect> faces;
        faceCascade.detectMultiScale(grayImage, faces, 1.1, 4, 0, cv::Size(96, 96));

        if (!faces.empty())
        {
            cv::Rect faceRect = faces[0];

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

            cv::rectangle(originalImage, faceRect, cv::Scalar(0, 255, 0), 2);
            cv::Mat faceCrop = grayImage(faceRect);

            cv::Mat blob;
            cv::dnn::blobFromImage(faceCrop, blob, 1.0/255.0, cv::Size(96, 96), cv::Scalar(), false, false);

            net.setInput(blob);
            cv::Mat output = net.forward();

            std::vector<float> predictions;
            for (int i = 0; i < output.cols; i++)
                predictions.push_back(output.at<float>(0, i));

            for (size_t i = 0; i < predictions.size(); i += 2)
            {
                float x_crop = predictions[i] * faceRect.width;
                float y_crop = predictions[i+1] * faceRect.height;

                float x_original = x_crop + faceRect.x;
                float y_original = y_crop + faceRect.y;

                cv::circle(originalImage, cv::Point(x_original, y_original), 3, cv::Scalar(0, 255, 0), -1);
            }
        }

        QImage qImg(originalImage.data, originalImage.cols, originalImage.rows, originalImage.step, QImage::Format_RGB888);
        ui->imageLabel->setPixmap(QPixmap::fromImage(qImg).scaled(ui->imageLabel->size(), Qt::KeepAspectRatio));

    } catch (const cv::Exception& e) {
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


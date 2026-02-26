#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <QFileDialog>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    ui->treeView->setRootIsDecorated(false);
    ui->treeView->setIndentation(0);
    ui->treeView->setHeaderHidden(true);

    PyProcess = new QProcess(this);
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


void MainWindow::processAndDisplayImage(const QString &imagePath)
{
    if(currentModelPath.isEmpty() || imagePath.isEmpty()) return;

    cv::dnn::Net net = cv::dnn::readNetFromONNX(currentModelPath.toStdString());
    cv::Mat image = cv::imread(imagePath.toStdString());
    if(image.empty()) return;

    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0/255, cv::Size(96, 96), cv::Scalar(), true, false);

    net.setInput(blob);
    cv::Mat output = net.forward();

    std::vector<float> predictions;
    for (int i = 0; i < output.cols; i++)
        predictions.push_back(output.at<float>(0, i));

    int originalW = image.cols;
    int originalH = image.rows;

    for (size_t i = 0; i < predictions.size(); i += 2)
    {
        float x = predictions[i] * originalW;
        float y = predictions[i+1] * originalH;
        cv::circle(image, cv::Point(x, y), 3, cv::Scalar(0, 255, 0), -1);
    }

    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    QImage qImg(image.data, image.cols, image.rows, image.step, QImage::Format_RGB888);

    ui->imageLabel->setPixmap(QPixmap::fromImage(qImg).scaled(ui->imageLabel->size(), Qt::KeepAspectRatio));
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
    if(currentImagePath.isEmpty()) return;
    processAndDisplayImage(currentImagePath);
}


void MainWindow::on_treeView_doubleClicked(const QModelIndex &index)
{
    QString path = fileModel->filePath(index);
    currentImagePath = path;
    processAndDisplayImage(currentImagePath);
}


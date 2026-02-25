#include "mainwindow.h"
#include "./ui_mainwindow.h"

#include <QFileDialog>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    PyProcess = new QProcess(this);
    connect(PyProcess, &QProcess::readyReadStandardOutput, this, &MainWindow::readPythonOutput);
    connect(PyProcess, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &MainWindow::onProcessFinished);

    ui->progressBar->setVisible(false);
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


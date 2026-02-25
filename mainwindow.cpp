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

    ui->progressBar->setVisible(false);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::readPythonOutput()
{

}

void MainWindow::on_btnTrainModel_clicked()
{
    QString csvPath = QFileDialog::getOpenFileName(this, "Select Dataset", "", "CSV Files (*.csv)");
    if(csvPath.isEmpty()) return;

    ui->progressBar->setVisible(true);
    ui->progressBar->setValue(0);

    QString targetAcc = QString::number(ui->targetAccSpinBox->value());

    QString targetAccFName = targetAcc;
    targetAccFName.replace(",", "");
    targetAccFName.replace(".", "");

    int counter = 1;
    QString modelPath = "model_" + targetAccFName + "_" + QString::number(counter) + ".onnx";

    while

    QStringList args;
    args << "train_model.py" << csvPath << targetAcc << "new_model.unnx";

    PyProcess->start("python", args);
}



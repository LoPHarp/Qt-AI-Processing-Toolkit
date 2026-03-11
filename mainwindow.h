#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QProcess>
#include <QTimer>
#include <QFileInfo>
#include <QFileSystemModel>

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_btnSelectModel_clicked();
    void on_btnSelectZNM1Model_clicked();
    void on_btnTrainModel_clicked();
    void on_btnTrainZNM1Model_clicked();

    void on_btnRedefinePoints_clicked();
    void on_btnSelectImage_clicked();
    void on_treeView_doubleClicked(const QModelIndex &index);

    void readPythonOutput();
    void onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void on_tabWidget_currentChanged(int index);

private:
    enum ProcessType {
        ProcessNone,
        ProcessKeypoints,
        ProcessZNM1
    };
    ProcessType currentProcessType = ProcessNone;

    Ui::MainWindow *ui;
    QProcess *PyProcess;
    QString currentModelPath;
    QString currentModelZNM1Path;
    QString currentImagePath;
    QFileSystemModel *fileModel;
    cv::CascadeClassifier faceCascade;

    std::vector<float> currentKeypoints;

    void predictEmotion();
    void setModelAsActive(const QString &path);
    void setModelZNM1NameLabel(const QString &path);
    void processAndDisplayImage(const QString &imagePath, bool applyJitter = false);
};
#endif // MAINWINDOW_H

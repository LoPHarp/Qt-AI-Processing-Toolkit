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

    void on_btnSelectZNM2Model_clicked();

    void on_btnTrainZNM2Model_clicked();

private:
    enum ProcessType {
        ProcessNone,
        ProcessKeypoints,
        ProcessZNM1,
        ProcessZNM2
    };
    ProcessType currentProcessType = ProcessNone;

    Ui::MainWindow *ui;
    QProcess *PyProcess;
    QString currentModelPath;
    QString currentModelZNM1Path;
    QString currentModelZNM2Path;
    QString currentImagePath;
    QFileSystemModel *fileModel;
    cv::CascadeClassifier faceCascade;
    cv::Mat currentFaceCrop;

    std::vector<float> currentKeypoints;

    void predictEmotion();
    void predictEmotionZNM2();
    void setModelAsActive(const QString &path);
    void setModelZNM1NameLabel(const QString &path);
    void setModelZNM2NameLabel(const QString &path);
    void processAndDisplayImage(const QString &imagePath, bool applyJitter = false);
};
#endif // MAINWINDOW_H

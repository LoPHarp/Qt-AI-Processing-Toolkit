#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QProcess>
#include <QTimer>
#include <QFileInfo>
#include <QFileSystemModel>

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
    void on_btnTrainModel_clicked();

    void on_btnSelectImage_clicked();
    void on_btnRedefinePoints_clicked();
    void on_treeView_doubleClicked(const QModelIndex &index);

    void readPythonOutput();
    void onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);

private:
    Ui::MainWindow *ui;
    QProcess *PyProcess;
    QString currentModelPath;

    void setModelAsActive(const QString &path);

    QString currentImagePath;
    QFileSystemModel *fileModel;

    void processAndDisplayImage(const QString &imagePath);
};
#endif // MAINWINDOW_H

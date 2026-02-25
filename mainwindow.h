#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QProcess>
#include <QTimer>
#include <QFileInfo>

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
    void on_btnTrainModel_clicked();
    void readPythonOutput();
    void onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);

    void on_btnSelectModel_clicked();

private:
    Ui::MainWindow *ui;
    QProcess *PyProcess;
    QString currentModelPath;

    void setModelAsActive(const QString &path);
};
#endif // MAINWINDOW_H

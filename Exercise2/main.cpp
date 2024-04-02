#include <iostream>
#include "Eigen/Eigen"

using namespace std;
using namespace Eigen;

//scrivo una funzione che verifichi che una matrice A sia non singolare
bool SolveSystem(const Matrix2d& A,
                 const Vector2d& b,
                 double& errRel_PALU,
                 double& errRel_QR,
                 double& detA,
                 double& condA)
{
    JacobiSVD<Matrix2d> svd(A);
    //calcolo il vettore dei valori singolari
    Vector2d singularValuesA = svd.singularValues();
    //calcolo il condizionemento della matrice A...
    condA = singularValuesA.maxCoeff() / singularValuesA.minCoeff();
    //...e il suo determinante
    detA = A.determinant();

    if(singularValuesA.minCoeff() < 1e-16)
    {
        errRel_PALU = -1;
        errRel_QR = -1;
        return false;
    }

    //definisco il vettore soluzione esatta dato dalla traccia
    Vector2d exactSolution;
    exactSolution << -1.0e+0, -1.0e+00;

    //calcolo la soluzione del sistema data dalla fattorizzazione PALU
    Vector2d x_PALU = A.fullPivLu().solve(b);

    //calcolo la soluzione del sistema data dalla fattorizzazione QR
    Vector2d x_QR = A.colPivHouseholderQr().solve(b);

    //calcolo gli errori relativi
    errRel_PALU = (exactSolution - x_PALU).norm() / exactSolution.norm();
    errRel_QR = (exactSolution - x_QR).norm() / exactSolution.norm();

    return true;
}


int main()
{
    //SISTEMA 1
    //inizializzo la matrice A1 e il vettore termine noto b1
    Matrix2d A1;
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
        8.320502943378437e-01, -9.992887623566787e-01;

    Vector2d b1;
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;

    //definisco le seguenti variabili:
    double errRel1_PALU, errRel1_QR, detA1, condA1;

    //verifico che la matrice A1 sia non singolare
    if (SolveSystem(A1,b1,errRel1_PALU, errRel1_QR, detA1, condA1))
        cout << scientific << "La matrice A1 è non singolare e gli errori relativi sono: " << errRel1_PALU << ", " << errRel1_QR << endl;
    else
        cout << "La matrice A1 è singolare." << endl;

    //calcolo la soluzione del sistema data dalla fattorizzazione PALU
    Vector2d x1_PALU = A1.fullPivLu().solve(b1);
    //calcolo la soluzione del sistema data dalla fattorizzazione QR
    Vector2d x1_QR = A1.colPivHouseholderQr().solve(b1);

    cout << "La soluzione del sistema 1 con la fattorizzazione PALU è: " << endl << x1_PALU << endl;
    cout << "La soluzione del sistema 1 con la fattorizzazione QR è: " << endl << x1_QR << endl;



    //SISTEMA 2
    //inizializzo la matrice A2 e il vettore termine noto b2
    Matrix2d A2;
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
        8.320502943378437e-01, -8.324762492991313e-01;

    Vector2d b2;
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;

    //definisco le seguenti variabili:
    double errRel2_PALU, errRel2_QR, detA2, condA2;

    //verifico che la matrice A2 sia non singolare
    if (SolveSystem(A2,b2,errRel2_PALU, errRel2_QR, detA2, condA2))
        cout << scientific << "La matrice A2 è non singolare e gli errori relativi sono: " << errRel2_PALU << ", " << errRel2_QR << endl;
    else
        cout << "La matrice A2 è singolare." << endl;

    //calcolo la soluzione del sistema data dalla fattorizzazione PALU
    Vector2d x2_PALU = A2.fullPivLu().solve(b2);
    //calcolo la soluzione del sistema data dalla fattorizzazione QR
    Vector2d x2_QR = A2.colPivHouseholderQr().solve(b2);

    cout << "La soluzione del sistema 2 con la fattorizzazione PALU è: " << endl << x2_PALU << endl;
    cout << "La soluzione del sistema 2 con la fattorizzazione QR è: " << endl << x2_QR << endl;



    //SISTEMA 3
    //inizializzo la matrice A3 e il vettore termine noto b3
    Matrix2d A3;
    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
        8.320502943378437e-01, -8.320502947645361e-01;

    Vector2d b3;
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;

    //definisco le seguenti variabili:
    double errRel3_PALU, errRel3_QR, detA3, condA3;

    //verifico che la matrice A3 sia non singolare
    if (SolveSystem(A3,b3,errRel3_PALU, errRel3_QR, detA3, condA3))
        cout << scientific << "La matrice A3 è non singolare e gli errori relativi sono: " << errRel3_PALU << ", " << errRel3_QR << endl;
    else
        cout << "La matrice A3 è singolare." << endl;

    //calcolo la soluzione del sistema data dalla fattorizzazione PALU
    Vector2d x3_PALU = A3.fullPivLu().solve(b3);
    //calcolo la soluzione del sistema data dalla fattorizzazione QR
    Vector2d x3_QR = A3.colPivHouseholderQr().solve(b3);

    cout << "La soluzione del sistema 3 con la fattorizzazione PALU è: " << endl << x3_PALU << endl;
    cout << "La soluzione del sistema 3 con la fattorizzazione QR è: " << endl << x3_QR << endl;

    return 0;
}



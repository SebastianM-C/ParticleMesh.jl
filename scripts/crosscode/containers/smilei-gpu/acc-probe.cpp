// Compiled to an object (not linked, not run) by the smilei-gpu build to ask nvc++ whether it accepts -gpu=cc89.
int main()
{
    double s = 0.0;
#pragma acc parallel loop reduction(+ : s)
    for (int i = 0; i < 64; i++) {
        s += i;
    }
    return s == 2016.0 ? 0 : 1;
}

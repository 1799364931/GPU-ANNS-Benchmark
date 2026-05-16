#include <stdio.h>
int main()
{
    int n, i;
    double sum = 0.0;
    scanf("%d", &n);
    if (n > 0)
    {
        for (i = 1; i <= n; i++)
        {
            sum += (2 * i - 1.0) / i;
        }
    }
    printf("%.2lf", sum);
    return 0;
}
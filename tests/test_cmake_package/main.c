#include <apg_sgm/c_api.h>
#include <stdio.h>

int main(void) {
    const char* ver = apg_version_string();
    printf("Consumer C app linked apg_sgm, version=%s\n", ver);
    return 0;
}

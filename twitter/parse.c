#include <stdio.h>
#include <string.h>
#define CHUNK 1024 /* read 1024 bytes at a time */

int main() {
    char buf[CHUNK];
    FILE *file;
    size_t nread;
    char p[CHUNK];

    file = fopen("twitter-corpus.csv", "r");
    if (file) {
        while ((nread = fread(buf, 1, sizeof buf, file)) > 0) {


            p = strtok(buf, ",");
            if(p) {
                printf("%s", p);
            }
            //fwrite(buf, 1, nread, stdout);
        }
        if (ferror(file)) {
            /* deal with error */
        }
        fclose(file);
    }
}

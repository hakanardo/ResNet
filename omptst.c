void main() {
  int maxthread, threadget;
  #pragma omp parallel
  {
    // be conservative, set number of real cores
    maxthread = omp_get_num_procs() / 2 - 1;
  }
  printf("%d\n", maxthread);
  int preprocess_threads = maxthread;
  #pragma omp parallel num_threads(preprocess_threads)
  {
    threadget = omp_get_num_threads();
  }
  printf("%d, %d\n", maxthread, threadget);
}
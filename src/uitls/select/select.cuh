
/*

*/

// each thread own N elements list, N is power of 2 , and list is in order aes
// lane0 own l[0] l[N] l[2*N]
// 0 1 2 3 | 0 1 2 3 | 0 1 2 3

template <typename K, int N> __device__ inline void merge(K k1[N], K k2[N]) {
  for (int i = 0; i < N; i++) {
    K ka = &k1[N - i - 1];
    K kb = &k2[i];

    K otherKb = __shfl_xor_sync(0xffff'ffff, ka, warpSize, warpSize);

    if (ka > otherKb) {
      ka = otherKb;
    }

    K otherKa = __shfl_xor_sync(0xffff'ffff, kb, warpSize, warpSize);

    if (kb < otherKa) {
      kb = otherKa;
    }
  }
}

template <typename K, int N, bool Low>
__device__ inline void bitonicStep(K k[N]) {
  for (int i = 0; i < N / 2; i++) {
    K &ka = k[i];
    K &kb = k[i + N / 2];

    if (ka < kb) {
      K t = ka;
      ka = kb;
      kb = t;
    }
  }
  
  {
    K newK[N / 2];

    for (int i = 0; i < N / 2; i++) {
      newK[i] = k[i];
    }

    bitonicStep<K, N / 2, true>(newK);

    for (int i = 0; i < N / 2; i++) {
      k[i] = newK[i];
    }
  }

  {
    K newK[N / 2];

    for (int i = 0; i < N / 2; i++) {
      newK[i] = k[i + N / 2];
    }

    bitonicStep<K, N / 2, true>(newK);

    for (int i = 0; i < N / 2; i++) {
      k[i + N / 2] = newK[i];
    }
  }
}
extern "C" {


/**
* @brief Check whether the number x_d is in an array y_d 
* @param x_d number to be checked
* @param y_d array to be checked
* @param N   length of array
* 
* @example1
* x_d = 2
* y_d = [1, 2, 3]
* N   = 3
* return true
* 
* @example2
* x_d = 4
* y_d = [1, 2, 3]
* N   = 3
* return false
**/
__device__ bool logic_in(const int x_d,
                         int *y_d,
                         const int N)
{
    for (int i = 0; i < N; ++ i) {
        if (x_d == y_d[i]) {
            return true;
        }
    }
    return false;
}

/**
@brief Finding all k-size subclusters
This method starts from (k-1) size subclusters and then add new vertices to existed
subclusters by iterating bonds.

@Note It's possible that multiple k-size subclusters in the output
are identical, so unique mechanisem needs to be applied later.

@param bonds (List[nbond * 2]): existed bonds in the graph
@param cluster_prev (List[nprev * sprev]): previous subclusters with (k - 1) size
@param cluster_new (List[npotential * (sprev + 1)]): new subclusters with k size
@param nbond: number of bonds in the graph
@param nprev: number of (k - 1) size subclusters
@param sprev: k - 1
@param index (int[0]): shared index to indicate the number of k size subclusters found
@param strong (bool): whether to find complete graph
**/
__global__ void find_subclusters(int* bonds,
                                 int* cluster_prev,
                                 int* cluster_new,
                                 const int nbond,
                                 const int nprev,
                                 const int sprev,
                                 int* index,
                                 bool strong
                                 )
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int icluster = bid * blockDim.x + tid;

    extern __shared__ int shared_bonds[];

    int *cluster_prev_l = NULL;
    if (icluster < nprev) {
        cluster_prev_l = (int *)malloc(sizeof(int) * sprev);
        for (int iprev = 0; iprev < sprev; ++ iprev) {
            // idx = icluster * sprev + iprev
            cluster_prev_l[iprev] = cluster_prev[icluster * sprev + iprev];
        }
    }

    const int iter = (nbond - 1) / blockDim.x + 1;
    for (int iter_id = 0; iter_id < iter; ++ iter_id) {
        const int bond_start_id = iter_id * blockDim.x;
        const int bond_id = bond_start_id + tid;
        __syncthreads();
        if (bond_id < nbond) {
            shared_bonds[2 * tid] = bonds[2 * bond_id];
            shared_bonds[2 * tid + 1] = bonds[2 * bond_id + 1];
        }
        else {
            shared_bonds[2 * tid] = 0;
            shared_bonds[2 * tid + 1] = 0;
        }
        __syncthreads();

        if (icluster < nprev) {
            for (int bond_idx = 0; bond_idx < blockDim.x; ++bond_idx) {
                int toadd = -1;
                if (logic_in(shared_bonds[2 * bond_idx], cluster_prev_l, sprev) && !logic_in(shared_bonds[2 * bond_idx + 1], cluster_prev_l, sprev)) {
                    toadd = shared_bonds[2 * bond_idx + 1];
                }
                else if (logic_in(shared_bonds[2 * bond_idx + 1], cluster_prev_l, sprev) && !logic_in(shared_bonds[2 * bond_idx], cluster_prev_l, sprev)) {
                    toadd = shared_bonds[2 * bond_idx];
                }

                if (strong && toadd != -1) {
                    bool valid = true;
                    for (int v = 0; v < sprev; ++v) {
                        bool bond_found = false;
                        for (int b = 0; b < nbond; ++b) {
                            if ((bonds[2 * b] == cluster_prev_l[v] && bonds[2 * b + 1] == toadd) || (bonds[2 * b + 1] == cluster_prev_l[v] && bonds[2 * b] == toadd)) {
                                bond_found = true;
                                break;
                            }
                        }
                        if (!bond_found) {
                            valid = false;
                            break;
                        }
                    }
                    if (!valid) {
                        toadd = -1;
                    }
                }

                // printf("shared_bonds: %d - %d\n", shared_bonds[2 * bond_idx], shared_bonds[2 * bond_idx + 1]);
                // printf("cluster_prev: %d - %d\n", cluster_prev_l[0], cluster_prev_l[1]);
                // printf("to add: %d\n", toadd);

                if (toadd != -1) {
                    int new_icluster = atomicAdd(index, 1);
                    const int nrow = new_icluster * (sprev + 1);
                    int insert_idx = sprev;
                    if (toadd < cluster_prev_l[0]) {
                        insert_idx = 0;
                    }
                    else if (toadd > cluster_prev_l[sprev - 1]) {
                        insert_idx = sprev;
                    }
                    else {
                        for (int i_v = 0; i_v < sprev - 1; ++ i_v) {
                            if (cluster_prev_l[i_v] < toadd && toadd < cluster_prev_l[i_v + 1]) {
                                insert_idx = i_v + 1;
                                break;
                            }
                        }
                    }
                    for (int i_v = 0; i_v < insert_idx; ++ i_v) {
                        cluster_new[nrow + i_v] = cluster_prev_l[i_v];
                    }
                    cluster_new[nrow + insert_idx] = toadd;
                    for (int i_v = insert_idx + 1; i_v < sprev + 1; ++ i_v) {
                        cluster_new[nrow + i_v] = cluster_prev_l[i_v - 1];
                    }
                }
            }
        }
    }
    if (cluster_prev_l != NULL) {
        free(cluster_prev_l);
    }
}


}

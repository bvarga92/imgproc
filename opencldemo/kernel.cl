void kernel vecadd(global const int* a, global const int* b, global int* c){
	int thId=get_global_id(0); //minden szal a neki megfelelo elemeket adja ossze
	c[thId]=a[thId]+b[thId];
}

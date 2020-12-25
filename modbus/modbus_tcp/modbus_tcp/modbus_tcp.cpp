#include <iostream>
#include "libmodbus/modbus.h"
#pragma comment(lib,"modbus.lib")  //这一步也可以通过Project->properties->linker-
//>input->additional additional dependencies添加用到的lib

using namespace std;
int main()
{
	modbus_t* mb;
	uint16_t tab_reg[3] = { 0 };
	uint8_t tab_reg1[1] = { 0 };
	
//	int nPort = 600;
	int nPort = 7930;
//	const char* chIp = "10.14.64.74";

	const char* chIp = "169.254.118.26";
//	const char* chIp = "172.20.10.5";
	mb = modbus_new_tcp(chIp, nPort);
	
//	mb = modbus_new_rtu("COM1", 9600, 'N', 8, 1);   //相同的端口只能同时打开一个 
	modbus_set_slave(mb, 1);  //设置modbus从机地址 

	modbus_connect(mb);
	//modbus_mapping_new()

	struct timeval t;
	t.tv_sec = 0;
	t.tv_usec = 1000000;   //设置modbus超时时间为1000毫秒 
	modbus_set_response_timeout(mb, (int)&t.tv_sec, (int)&t.tv_usec);

		//tab_reg[0] = 1;
		//tab_reg[1] = 2;
		//tab_reg[2] = 3;
	cout << tab_reg[0] << " " << tab_reg[1] << " " << tab_reg[2] << endl;
//	while (1) {
//		cout << tab_reg1[0] << endl;
////		modbus_read_input_bits(mb, 3, 1, tab_reg1);
//		modbus_read_bits(mb, 3, 1, tab_reg1);
//
//		if (tab_reg1[0] == 1) break;
//	}
	int i = 0;
	while(1) {
		//tab_reg[0] = i;
		//tab_reg[1] = i*2;
		//tab_reg[2] = i*3;
		tab_reg[0] = 2;
		tab_reg[1] = 3;
		tab_reg[2] = 4;
		i++;
		int nRet = modbus_write_registers(mb, 40001, 3, tab_reg);
		cout << tab_reg[0] <<" "<< tab_reg[1]<<" " << tab_reg[2] << endl;
		Sleep(1000);

	}

//	}




//	system("pause");
	return 0;
}
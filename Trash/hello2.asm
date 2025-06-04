
hello:     file format elf32-i386


Disassembly of section .text:

08049000 <_start>:
 8049000:	b8 04 00 00 00       	mov    $0x4,%eax
 8049005:	bb 01 00 00 00       	mov    $0x1,%ebx
 804900a:	b9 00 a0 04 08       	mov    $0x804a000,%ecx
 804900f:	ba 0e 00 00 00       	mov    $0xe,%edx
 8049014:	cd 80                	int    $0x80
 8049016:	b8 01 00 00 00       	mov    $0x1,%eax
 804901b:	31 db                	xor    %ebx,%ebx
 804901d:	cd 80                	int    $0x80

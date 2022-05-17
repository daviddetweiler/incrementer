.global run_null_test

.text
run_null_test:
	xor		%r10, %r10

begin:
	cmp		%rdi, %r10
	je		done

	// Address index
	mov		%r10, %r11
	shl		$3, %r11
	add		%rdx, %r11

	// Address cacheline
	mov		(%r11), %r11
	shl		$5, %r11
	add		%rsi, %r11

	incq	(%r11)
	
	inc		%r10
	jmp		begin

done:	
	ret

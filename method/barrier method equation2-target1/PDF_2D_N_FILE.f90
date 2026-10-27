program main
    implicit none
    integer     ::  e,i,hnum,vnum,bnc,N_File
    real*8     ::  a,b,c,d,hgap,vgap,time_begin,time_end

    common  /group1/ hnum,vnum,bnc
    common  /group2/ a,b,c,d,hgap,vgap
    common  /group3/ N_File

    write(*,*)"***************************************************************"
    write(*,*)"*                                                             *"
    write(*,*)"*      Monte Carlo numerical simulation on phase plane         *"
    write(*,*)"*      The file names should be: pdf_001,pdf_002.....         *"                                                 
    write(*,*)"*                                                             *"
    write(*,*)"***************************************************************"

    write(*,*)
    write(*,*)"Input the number of FILES: "
    read(*,*)N_File
    write(*,*)
    write(*,*)"Input the partition scales hnum and vnum: "
    read(*,*)hnum,vnum
    write(*,*)
    write(*,*)"Input the basin boundary [a b c d]: "
    read(*,*)a,b,c,d

    bnc     =   hnum*vnum
    hgap    =   (b-a)/hnum
    vgap    =   (d-c)/vnum

    call cpu_time(time_begin)
    call pdf()
    call cpu_time(time_end)

    write(*,*)
    write(*,*)"The time is: ",time_end-time_begin

end program

subroutine coordinate(n,aa,bb,cc,dd)
    implicit none
    real*8      ::  aa,bb,cc,dd
    integer     ::  n,u1,u2,I1,I2,hnum,vnum,bnc
    real*8     ::  a,b,c,d,hgap,vgap

    common  /group1/ hnum,vnum,bnc
    common  /group2/ a,b,c,d,hgap,vgap

    u1 = mod(n,hnum)
    u2 = int(n/hnum) + 1

    if(u1 == 0)then
        I1 = hnum
        I2 = u2 - 1
    else
        I1 = u1
        I2 = u2
    endif

    aa = a + (I1-1.0)*hgap
    bb = a + I1*hgap
    cc = c + (I2-1.0)*vgap
    dd = c + I2*vgap
    return
end subroutine

subroutine point_cell(n,x,y)
    implicit none
    real*8      ::  x,y
    integer     ::  i,j,n,hnum,vnum,bnc
    real*8     ::  a,b,c,d,hgap,vgap

    common  /group1/ hnum,vnum,bnc
    common  /group2/ a,b,c,d,hgap,vgap

    ! Check if within boundaries
    if(x >= a .and. x <= b .and. y >= c .and. y <= d) then
        ! Calculate indices
        i = int((x-a)/hgap) + 1
        j = int((y-c)/vgap) + 1

        ! Handle edge cases
        if (i > hnum) i = hnum
        if (j > vnum) j = vnum

        n = (j-1)*hnum + i
    else
        n = 0
    endif

    return
end subroutine

subroutine pdf()
    implicit none
    integer             ::  error, i, n, total, hnum, vnum, bnc, N_File
    integer             ::  error_open
    real*8              ::  a1,b1,c1,d1,x_val,y_val
    real*8              ::  a,b,c,d,hgap,vgap
    real*8, allocatable ::  temp(:)
    real*8              ::  px(0:hnum-1), py(0:vnum-1)
    character(len=35)   ::  filename1
    character(len=3)   ::  str
    logical             ::  file_exists

    common  /group1/ hnum,vnum,bnc
    common  /group2/ a,b,c,d,hgap,vgap
    common  /group3/ N_File
    
    allocate(temp(bnc))

    ! Open output files
    open(unit=100,file='p(x,y).txt')
    open(unit=200,file='p(x,0).txt')
    open(unit=300,file='p(0,y).txt')

    write(100,*)"title = 'pdf' "
    write(100,*)"variables = 'x', 'y', 'z' "
    write(100,*)"zone i=",hnum,", j=",vnum,", f=point"

    temp = 0
    total = 0
    
    do i = 1, N_File
        
        write(str,'(I3.3)') i
        filename1 = 'pdf_'//str//'.txt'
        
        ! Check if file exists before opening
        inquire(file=filename1, exist=file_exists)
        if (.not. file_exists) then
            write(*,*) 'Warning: File does not exist - ', trim(filename1)
            cycle
        endif
        
        open(unit=10,file=filename1,iostat=error_open)
        if(error_open /= 0) then
            write(*,*) 'Warning: Cannot open ', trim(filename1), ' (iostat=', error_open, ')'
            cycle
        endif
        
        write(*,*) 'Reading file: ', trim(filename1)
        
        ! Read data until end of file
        do while(.true.)
            read(10,*,iostat=error) x_val, y_val
            if(error /= 0) exit
            
            call point_cell(n,x_val,y_val)
            if(n /= 0) temp(n) = temp(n) + 1
            total = total + 1
            if(mod(total, 25000) == 0) then
                write(*,*) 'Points processed so far: ', total
            endif
        enddo
        
        close(10)
        write(*,*) 'Finished reading: ', trim(filename1)
        
    enddo

    write(*,*) 'Total points: ', total
    
    px = 0
    py = 0
    
    ! Calculate probability density and marginal distributions
    do i = 1, bnc
        call coordinate(i,a1,b1,c1,d1)
        x_val = (a1+b1)/2.
        y_val = (c1+d1)/2.
        write(100,*) x_val, y_val, temp(i)/total/hgap/vgap

        px(mod((i-1),hnum)) = px(mod((i-1),hnum)) + temp(i)/total
        py(int((i-1)/hnum)) = py(int((i-1)/hnum)) + temp(i)/total

    enddo
    
    ! Write marginal distributions
    do i = 0, hnum-1
        write(200,*) a + hgap*(i+0.5), px(i)/hgap/sum(px)
    enddo
    
    do i = 0, vnum-1
        write(300,*) c + vgap*(i+0.5), py(i)/vgap/sum(py)
    enddo
    
    close(100)
    close(200)
    close(300)
    deallocate(temp)
    
end subroutine
module spldlt_utils_mod
  use spral_ssids_datatypes
  implicit none

  interface sort
     module procedure sorti, sortl
  end interface sort

contains

  ! sort routines adapted from SPRAL

  !************************************************************************
  
  !
  !   Sort an integer array by heapsort into ascending order.
  !
  subroutine sortl( array, n, map, val )
    integer, intent(in) :: n       ! Size of array to be sorted
    integer(long), dimension(n), intent(inout) :: array ! Array to be sorted
    integer, dimension(n), optional, intent(inout) :: map
    real(wp), dimension(n), optional, intent(inout) :: val ! Apply same
    ! permutation to val

    integer :: i
    integer(long) :: temp
    integer       :: tempi
    real(wp) :: vtemp
    integer :: root

    if(n.le.1) return ! nothing to do

    !
    ! Turn array into a heap with largest element on top (as this will be pushed
    ! on the end of the array in the next phase)
    !
    ! We start at the bottom of the heap (i.e. 3 above) and work our way
    ! upwards ensuring the new "root" of each subtree is in the correct
    ! location
    root = n / 2
    do root = root, 1, -1
       call pushdownl(root, n, array, val=val, map=map)
    end do

    !
    ! Repeatedly take the largest value and swap it to the back of the array
    ! then repeat above code to sort the array
    !
    do i = n, 2, -1
       ! swap a(i) and head of heap a(1)
       temp = array(1)
       array(1) = array(i)
       array(i) = temp
       if(present(val)) then
          vtemp = val(1)
          val(1) = val(i)
          val(i) = vtemp
       endif
       if(present(map)) then
          tempi = map(1)
          map(1) = map(i)
          map(i) = tempi
       endif
       call pushdownl(1,i-1, array, val=val, map=map)
    end do
  end subroutine sortl


  !****************************************
  
  ! This subroutine will assume everything below head is a heap and will
  ! push head downwards into the correct location for it
  subroutine pushdownl(root, last, array, val, map)
    integer, intent(in) :: root
    integer, intent(in) :: last
    integer(long), dimension(last), intent(inout) :: array
    real(wp), dimension(last), optional, intent(inout) :: val
    integer, dimension(last), optional, intent(inout) :: map

    integer :: insert ! current insert position
    integer :: test ! current position to test
    integer(long) :: root_idx ! value of array(root) at start of iteration
    real(wp) :: root_val ! value of val(root) at start of iteration
    integer :: root_map ! value of map(root) at start of iteration

    ! NB a heap is a (partial) binary tree with the property that given a
    ! parent and a child, array(child)>=array(parent).
    ! If we label as
    !                      1
    !                    /   \
    !                   2     3
    !                  / \   / \
    !                 4   5 6   7
    ! Then node i has nodes 2*i and 2*i+1 as its children

    if(present(val) .and. present(map)) then ! both val and map
       root_idx = array(root)
       root_val = val(root)
       root_map = map(root)
       insert = root
       test = 2*insert
       do while(test.le.last)
          ! First check for largest child branch to descend
          if(test.ne.last) then
             if(array(test+1).gt.array(test)) test = test + 1
          endif
          if(array(test).le.root_idx) exit ! root gets tested here
          ! Otherwise, move on to next level down, percolating value up
          array(insert) = array(test);
          val(insert) = val(test);
          map(insert) = map(test)
          insert = test
          test = 2*insert
       end do
       ! Finally drop root value into location found
       array(insert) = root_idx
       val(insert) = root_val
       map(insert) = root_map
    elseif(present(val)) then ! val only, not map
       root_idx = array(root)
       root_val = val(root)
       insert = root
       test = 2*insert
       do while(test.le.last)
          ! First check for largest child branch to descend
          if(test.ne.last) then
             if(array(test+1).gt.array(test)) test = test + 1
          endif
          if(array(test).le.root_idx) exit ! root gets tested here
          ! Otherwise, move on to next level down, percolating value up
          array(insert) = array(test)
          val(insert) = val(test)
          insert = test
          test = 2*insert
       end do
       ! Finally drop root value into location found
       array(insert) = root_idx
       val(insert) = root_val
    elseif(present(map)) then ! map only, not val
       root_idx = array(root)
       root_map = map(root)
       insert = root
       test = 2*insert
       do while(test.le.last)
          ! First check for largest child branch to descend
          if(test.ne.last) then
             if(array(test+1).gt.array(test)) test = test + 1
          endif
          if(array(test).le.root_idx) exit ! root gets tested here
          ! Otherwise, move on to next level down, percolating mapue up
          array(insert) = array(test)
          map(insert) = map(test)
          insert = test
          test = 2*insert
       end do
       ! Finally drop root mapue into location found
       array(insert) = root_idx
       map(insert) = root_map
    else ! neither map nor val
       root_idx = array(root)
       insert = root
       test = 2*insert
       do while(test.le.last)
          ! First check for largest child branch to descend
          if(test.ne.last) then
             if(array(test+1).gt.array(test)) test = test + 1
          endif
          if(array(test).le.root_idx) exit ! root gets tested here
          ! Otherwise, move on to next level down, percolating value up
          array(insert) = array(test)
          insert = test
          test = 2*insert
       end do
       ! Finally drop root value into location found
       array(insert) = root_idx
    endif

  end subroutine pushdownl

  !************************************************************************
  
  !
  !   Sort an integer array by heapsort into ascending order.
  !
  subroutine sorti( array, n, map, val )
    integer, intent(in) :: n       ! Size of array to be sorted
    integer, dimension(n), intent(inout) :: array ! Array to be sorted
    integer, dimension(n), optional, intent(inout) :: map
    real(wp), dimension(n), optional, intent(inout) :: val ! Apply same
    ! permutation to val

    integer :: i
    integer :: temp
    real(wp) :: vtemp
    integer :: root

    if(n.le.1) return ! nothing to do

    !
    ! Turn array into a heap with largest element on top (as this will be pushed
    ! on the end of the array in the next phase)
    !
    ! We start at the bottom of the heap (i.e. 3 above) and work our way
    ! upwards ensuring the new "root" of each subtree is in the correct
    ! location
    root = n / 2
    do root = root, 1, -1
       call pushdowni(root, n, array, val=val, map=map)
    end do

    !
    ! Repeatedly take the largest value and swap it to the back of the array
    ! then repeat above code to sort the array
    !
    do i = n, 2, -1
       ! swap a(i) and head of heap a(1)
       temp = array(1)
       array(1) = array(i)
       array(i) = temp
       if(present(val)) then
          vtemp = val(1)
          val(1) = val(i)
          val(i) = vtemp
       endif
       if(present(map)) then
          temp = map(1)
          map(1) = map(i)
          map(i) = temp
       endif
       call pushdowni(1,i-1, array, val=val, map=map)
    end do
  end subroutine sorti


  !****************************************
  
  ! This subroutine will assume everything below head is a heap and will
  ! push head downwards into the correct location for it
  subroutine pushdowni(root, last, array, val, map)
    integer, intent(in) :: root
    integer, intent(in) :: last
    integer, dimension(last), intent(inout) :: array
    real(wp), dimension(last), optional, intent(inout) :: val
    integer, dimension(last), optional, intent(inout) :: map

    integer :: insert ! current insert position
    integer :: test ! current position to test
    integer :: root_idx ! value of array(root) at start of iteration
    real(wp) :: root_val ! value of val(root) at start of iteration
    integer :: root_map ! value of map(root) at start of iteration

    ! NB a heap is a (partial) binary tree with the property that given a
    ! parent and a child, array(child)>=array(parent).
    ! If we label as
    !                      1
    !                    /   \
    !                   2     3
    !                  / \   / \
    !                 4   5 6   7
    ! Then node i has nodes 2*i and 2*i+1 as its children

    if(present(val) .and. present(map)) then ! both val and map
       root_idx = array(root)
       root_val = val(root)
       root_map = map(root)
       insert = root
       test = 2*insert
       do while(test.le.last)
          ! First check for largest child branch to descend
          if(test.ne.last) then
             if(array(test+1).gt.array(test)) test = test + 1
          endif
          if(array(test).le.root_idx) exit ! root gets tested here
          ! Otherwise, move on to next level down, percolating value up
          array(insert) = array(test);
          val(insert) = val(test);
          map(insert) = map(test)
          insert = test
          test = 2*insert
       end do
       ! Finally drop root value into location found
       array(insert) = root_idx
       val(insert) = root_val
       map(insert) = root_map
    elseif(present(val)) then ! val only, not map
       root_idx = array(root)
       root_val = val(root)
       insert = root
       test = 2*insert
       do while(test.le.last)
          ! First check for largest child branch to descend
          if(test.ne.last) then
             if(array(test+1).gt.array(test)) test = test + 1
          endif
          if(array(test).le.root_idx) exit ! root gets tested here
          ! Otherwise, move on to next level down, percolating value up
          array(insert) = array(test)
          val(insert) = val(test)
          insert = test
          test = 2*insert
       end do
       ! Finally drop root value into location found
       array(insert) = root_idx
       val(insert) = root_val
    elseif(present(map)) then ! map only, not val
       root_idx = array(root)
       root_map = map(root)
       insert = root
       test = 2*insert
       do while(test.le.last)
          ! First check for largest child branch to descend
          if(test.ne.last) then
             if(array(test+1).gt.array(test)) test = test + 1
          endif
          if(array(test).le.root_idx) exit ! root gets tested here
          ! Otherwise, move on to next level down, percolating mapue up
          array(insert) = array(test)
          map(insert) = map(test)
          insert = test
          test = 2*insert
       end do
       ! Finally drop root mapue into location found
       array(insert) = root_idx
       map(insert) = root_map
    else ! neither map nor val
       root_idx = array(root)
       insert = root
       test = 2*insert
       do while(test.le.last)
          ! First check for largest child branch to descend
          if(test.ne.last) then
             if(array(test+1).gt.array(test)) test = test + 1
          endif
          if(array(test).le.root_idx) exit ! root gets tested here
          ! Otherwise, move on to next level down, percolating value up
          array(insert) = array(test)
          insert = test
          test = 2*insert
       end do
       ! Finally drop root value into location found
       array(insert) = root_idx
    endif

  end subroutine pushdowni

end module spldlt_utils_mod

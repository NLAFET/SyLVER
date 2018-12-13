#pragma once

// STD
#include <cassert>

namespace spldlt {

   // Block structure for unsymmetric matrices
   //
   // TODO Create generic block type to cover sym posdef, indef and
   // unsym matrices
   template<typename T>
   class BlockUnsym {
   public:
      BlockUnsym()
         : m(0), n(0), a(nullptr), lda(0), b(nullptr), ldb(0), mb_(0), nb_(0)
      {}

      BlockUnsym(int i, int j, int m, int n, T* a, int lda)
         : i(i), j(j), m(m), n(n), a(a), lda(lda), mb_(0), nb_(0)
      {}

      BlockUnsym(int i, int j, int m, int n, T* a, int lda, int mb, int nb,
                 T* b, int ldb)
         : i(i), j(j), m(m), n(n), a(a), lda(lda), mb_(mb), nb_(nb), b(b),
           ldb(ldb)
      {
         assert(mb_ <= m);
         assert(nb_ < n);
      }

      inline int get_mb() { return mb_;}
      inline int get_nb() { return nb_;}

      /// @brief Get number of rows in a
      inline int get_ma() {
         return m;
      }

      /// @brief Get number of columns in a
      inline int get_na() {
         return n-get_nb();
      }

      int i; // block row index
      int j; // block column index
      int m; // Number of rows in block
      int n; // Number of columns in block
      int lda; // leading dimension of underlying storage
      T* a; // pointer to underlying matrix storage
      // Note that the block might be slip in two different memory
      // areas (e.g. one for factor L and another one for U). In this
      // case we use b to point on the reminder:
      int ldb; // leading dimension of underlying storage
      T* b; // pointer to underlying matrix storage  
   private:
      int mb_;
      int nb_;
      
      // case 1) Block is in a single memory location (lcol or ucol)
      // e.g.
     
      // +-------+
      // |       |
      // |   a   | 
      // |       |
      // +-------+
      //

      // case 2) Block split between 2 memory locations(lcol and ucol)
      // e.g.
      
      // +---+---+
      // |   |   |
      // | a | b | 
      // |   |   |
      // +---+---+
      //
      // or
      // 
      // +---+---+  ---   ---
      // |   | b |   | mb  |
      // | a +---+  ---    | m
      // |   |             | 
      // +---+            ---
      
   };
   
} // namespaces spldlt

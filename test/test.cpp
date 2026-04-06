#include <iostream>

using namespace std;

struct ListNode{
  int val;
  ListNode* next;
};

ListNode* createNode(int val){
  ListNode* node = new ListNode();
  ListNode* cur = node;
  for (int i=0; i < val; i++){
    cur->val = i;
    cur->next = new ListNode();
    cur = cur->next;
  }
  return node;
}

void printList(ListNode* node){
  while (node != NULL){
    cout << node->val << " ";
    node = node->next;
  }
}

ListNode* reverseList(ListNode* node){
  ListNode* prev = NULL;
  ListNode* cur = node;
  while (cur != NULL){
    ListNode* tmp = cur->next;
    cur->next = prev;

    prev = cur;
    cur = tmp;
  }
  return prev;
}

int main(){
  ListNode* node = createNode(10);
  ListNode* reversed = reserveList(node);
  printList(reverseed);
  return 0;
}